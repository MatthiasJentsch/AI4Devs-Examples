// agent.mts

import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

import 'dotenv/config';

// Define the tools for the agent to use
const agentTools = [new TavilySearch({ maxResults: 3 })];
const agentModel = new ChatOpenAI({ model: "gpt-4.1-mini", temperature: 0 });

// Initialize memory to persist state between graph runs
const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
  llm: agentModel,
  tools: agentTools,
  checkpointSaver: agentCheckpointer,
});

// Now it's time to use!
const agentFinalState = await agent.invoke(
  { messages: [new HumanMessage("Wie ist das Wetter gerade in Regensburg?")] },
  { configurable: { thread_id: "42" } },
);

console.log(
  agentFinalState.messages[agentFinalState.messages.length - 1].content,
);

const agentNextState = await agent.invoke(
  { messages: [new HumanMessage("Wie ist das Wetter in München?")] },
  { configurable: { thread_id: "42" } },
);

console.log(
  agentNextState.messages[agentNextState.messages.length - 1].content,
);

import { writeFileSync } from "node:fs";

const graph = await agent.getGraphAsync();
const image = await graph.drawMermaidPng();
const arrayBuffer = await image.arrayBuffer();

const filePath = "./graphState.png";
writeFileSync(filePath, new Uint8Array(arrayBuffer));