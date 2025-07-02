import { Annotation } from "@langchain/langgraph";
import { SYSTEM_PROMPT_TEMPLATE } from "./prompts.js";
import { RunnableConfig } from "@langchain/core/runnables";

export const ConfigurationSchema = Annotation.Root({
    /**
     * The name of the language model to be used by the agent (e.g., "gpt-4o-mini", "gpt-4.1-mini").
     */
    model_name: Annotation<string>,
    /**
     * The provider of the language model (e.g., "openai", "anthropic").
     */
    model_provider: Annotation<string>,
    /**
     * The temperature to be used by the agent.
     */
    temperature: Annotation<number>,
    /**
     * The system prompt to be used by the agent. The term {system_time} will be replaced with the current date and time.
     */
    system_prompt: Annotation<string>,
});

export function ensureConfiguration(
  config: RunnableConfig,
): typeof ConfigurationSchema.State {
  /**
   * Ensure the defaults are populated.
   */
  const configurable = config.configurable ?? {};
  return {
    model_name: configurable.model_name ?? "gpt-4o-mini",
    model_provider: configurable.model_provider ?? "openai",
    temperature: configurable.temperature ?? 0,
    system_prompt:
      configurable.system_prompt ?? SYSTEM_PROMPT_TEMPLATE,
  };
}
