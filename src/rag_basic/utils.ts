import { BaseMessage } from "@langchain/core/messages";
import { Document } from "langchain/document";

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { initChatModel } from "langchain/chat_models/universal";

import { ChatOpenAI } from "@langchain/openai";
import { InferenceClient } from "@huggingface/inference";

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

/*export async function loadChatModel(
  provider: string,
  model: string,
): Promise<BaseChatModel> {
  return await initChatModel(model, { modelProvider: provider });
}*/

// Schema ist optional, wird bei withStructuredOutput benutzt
export async function loadChatModel(provider, modelName) {
  if (provider === "openai") {
    return new ChatOpenAI({
      modelName,
      temperature: 0.7,
    });
  }

  if (provider === "huggingface") {
    const apiKey = process.env.HF_API_KEY;
    if (!apiKey) {
      throw new Error("HF_API_KEY nicht gesetzt (in .env oder als Umgebungsvariable)");
    }

    const hf = new InferenceClient(apiKey);

    // Wir erstellen ein einfaches Wrapper-Objekt, das sich wie LangChain-Modelle verhält
    const wrapper = {
      invoke: async (messages) => {
        const prompt = messages.map((m) => m.content).join("\n");
        const result = await hf.textGeneration({
          model: modelName,
          inputs: prompt,
          parameters: {
            temperature: 0.7,
            max_new_tokens: 500,
            return_full_text: false,
          },
        });

        const output = Array.isArray(result) ? result[0].generated_text : result.generated_text;

        return {
          role: "assistant",
          content: output.trim(),
        };
      },

      withStructuredOutput(schema) {
        return {
          invoke: async (messages) => {
            const raw = await wrapper.invoke(messages);
            const parsed = schema.safeParse({ query: raw.content });
            if (!parsed.success) {
              throw new Error("Parsing failed: " + JSON.stringify(parsed.error));
            }
            return parsed.data;
          },
        };
      },
    };

    return wrapper;
  }

  throw new Error(`Unsupported provider: ${provider}`);
}
