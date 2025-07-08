import { BaseMessage } from "@langchain/core/messages";
import { Document } from "langchain/document";

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { initChatModel } from "langchain/chat_models/universal";

export function formatDoc(doc: Document): string {
  const metadata = doc.metadata || {};
  const meta = Object.entries(metadata)
    .map(([k, v]) => ` ${k}=${v}`)
    .join("");
  const metaStr = meta ? ` ${meta}` : "";

  return `<document${metaStr}>\n${doc.pageContent}\n</document>`;
}

export function formatDocs(docs?: Document[]): string {
  /**Format a list of documents as XML. */
  if (!docs || docs.length === 0) {
    return "<documents></documents>";
  }
  const formatted = docs.map(formatDoc).join("\n");
  return `<documents>\n${formatted}\n</documents>`;
}

export async function loadChatModel(
  provider: string,
  model: string,
): Promise<BaseChatModel> {
  return await initChatModel(model, { modelProvider: provider });
}