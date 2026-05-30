use tokio::sync::watch;

use super::Session;
use crate::message::{Message, ToolCall};
use crate::tool::{self, ShellImpact};

pub(super) enum AutoPermissionDecision {
    Approved,
    Denied {
        message: String,
    },
    AskUser {
        summary: String,
        diff: Option<String>,
        explanation: Option<String>,
        impact: ShellImpact,
    },
    Cancelled,
}

impl Session {
    pub(super) async fn reviewAutoPermission(
        &mut self,
        call: &ToolCall,
        action: &tool::ToolAction,
        summary: &str,
        reviewHistory: &[Message],
        cancelRx: &mut watch::Receiver<bool>,
    ) -> AutoPermissionDecision {
        let diff = tool::diffPreview(action);
        let explanation =
            crate::permissions::toolExplanation(action).map(std::string::ToString::to_string);
        let impact = crate::permissions::toolImpact(action);
        let meta = crate::auto_review::permissionMeta(&call.function.arguments);
        let actionHash =
            crate::auto_review::actionHash(&call.function.name, &call.function.arguments);

        if meta.raiseToUser {
            return match self.autoReviewTickets.remove(&actionHash) {
                Some(ticket) => {
                    let mut raisedExplanation = String::new();
                    if let Some(ref prior) = explanation {
                        raisedExplanation.push_str(prior);
                        raisedExplanation.push_str("\n\n");
                    }
                    raisedExplanation
                        .push_str("Auto reviewer allowed escalation for this exact action.");
                    if !ticket.reason.is_empty() {
                        raisedExplanation.push_str("\nReviewer reason: ");
                        raisedExplanation.push_str(&ticket.reason);
                    }
                    if !ticket.messageToAgent.is_empty() {
                        raisedExplanation.push_str("\nReviewer message: ");
                        raisedExplanation.push_str(&ticket.messageToAgent);
                    }
                    if let Some(ref reason) = meta.raiseReason {
                        raisedExplanation.push_str("\nAgent raise reason: ");
                        raisedExplanation.push_str(reason);
                    }
                    AutoPermissionDecision::AskUser {
                        summary: format!("{summary} (auto-review escalation)"),
                        diff,
                        explanation: Some(raisedExplanation),
                        impact,
                    }
                }
                None => AutoPermissionDecision::Denied {
                    message: "raiseToUser was set, but there is no active auto-review raise ticket for this exact action. Continue without asking the user, or retry without raiseToUser to request a fresh auto review.".into(),
                },
            };
        }

        let client = self.client.clone();
        let reviewInput = crate::auto_review::ReviewInput {
            toolCallId: call.id.clone(),
            toolName: call.function.name.clone(),
            summary: summary.to_string(),
            args: call.function.arguments.clone(),
            impact: impact.clone(),
            explanation: explanation.clone(),
            diff: diff.clone(),
        };
        let reviewResult = tokio::select! {
            review = crate::auto_review::review(
                &client,
                reviewHistory,
                reviewInput,
            ) => review,
            _ = cancelRx.changed() => {
                tracing::info!("cancelled during auto-review");
                return AutoPermissionDecision::Cancelled;
            }
        };

        match reviewResult {
            Ok(review) if review.allowed() => AutoPermissionDecision::Approved,
            Ok(review) => {
                if review.raiseAllowed() {
                    self.autoReviewTickets
                        .insert(actionHash, review.raiseTicket());
                }
                AutoPermissionDecision::Denied {
                    message: review.denialToolResult(),
                }
            }
            Err(e) => {
                let fallbackExplanation = {
                    let mut msg = explanation.unwrap_or_default();
                    if !msg.is_empty() {
                        msg.push_str("\n\n");
                    }
                    msg.push_str("Auto reviewer failed; falling back to user approval: ");
                    msg.push_str(&e.to_string());
                    Some(msg)
                };
                AutoPermissionDecision::AskUser {
                    summary: format!("{summary} (auto-review fallback)"),
                    diff,
                    explanation: fallbackExplanation,
                    impact,
                }
            }
        }
    }
}
