package com.mycompany.app.service;

import java.io.IOException;
import java.util.Collections;
import java.util.UUID;

import org.springframework.stereotype.Service;

import com.mycompany.app.helpers.VertexAiClient;
import com.mycompany.app.model.openai.ChatCompletionRequest;
import com.mycompany.app.model.openai.ChatCompletionResponse;
import com.mycompany.app.model.openai.CompletionRequest;
import com.mycompany.app.model.openai.CompletionResponse;

@Service
public class AiService {

    private final VertexAiClient vertexAiClient;

    public AiService(VertexAiClient vertexAiClient) {
        this.vertexAiClient = vertexAiClient;
    }

    /**
     * Creates a chat completion using Vertex AI, with a request and response format
     * compatible with the OpenAI API.
     *
     * @param request The chat completion request.
     * @return A chat completion response.
     */
    public ChatCompletionResponse createChatCompletion(ChatCompletionRequest request) {
        // For simplicity, we'll take the content of the last user message to form the prompt.
        // A more sophisticated approach might handle roles and conversation history differently.
        String prompt = request.getMessages().stream()
                .filter(m -> "user".equalsIgnoreCase(m.getRole()))
                .map(ChatCompletionRequest.Message::getContent)
                .reduce((first, second) -> second)
                .orElse("");

        String responseText;
        try {
            responseText = vertexAiClient.prompt(prompt);
        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt(); // Preserve the interrupted status
            // In a real application, you'd want more robust error handling,
            // possibly with a @ControllerAdvice.
            throw new RuntimeException("Error calling Vertex AI model", e);
        }

        // Construct an OpenAI-compatible response object.
        ChatCompletionResponse response = new ChatCompletionResponse();
        response.setId("chatcmpl-" + UUID.randomUUID().toString());
        response.setObject("chat.completion");
        response.setCreated(System.currentTimeMillis() / 1000L);
        response.setModel(request.getModel()); // Echo back the requested model

        ChatCompletionResponse.Choice choice = new ChatCompletionResponse.Choice();
        choice.setIndex(0);
        choice.setMessage(new ChatCompletionResponse.Message("assistant", responseText));
        choice.setFinishReason("stop");

        response.setChoices(Collections.singletonList(choice));
        return response;
    }

    /**
     * Creates a completion using Vertex AI, with a request and response format
     * compatible with the legacy OpenAI Completions API.
     *
     * @param request The completion request.
     * @return A completion response.
     */
    public CompletionResponse createCompletion(CompletionRequest request) {
        String prompt = request.getPrompt();

        String responseText;
        try {
            responseText = vertexAiClient.prompt(prompt);
        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt(); // Preserve the interrupted status
            // In a real application, you'd want more robust error handling.
            throw new RuntimeException("Error calling Vertex AI model", e);
        }

        // Construct an OpenAI-compatible response object.
        CompletionResponse response = new CompletionResponse();
        response.setId("cmpl-" + UUID.randomUUID().toString());
        response.setObject("text_completion");
        response.setCreated(System.currentTimeMillis() / 1000L);
        response.setModel(request.getModel()); // Echo back the requested model

        CompletionResponse.Choice choice = new CompletionResponse.Choice();
        choice.setIndex(0);
        choice.setText(responseText);
        choice.setFinishReason("stop");

        response.setChoices(Collections.singletonList(choice));
        return response;
    }
}
