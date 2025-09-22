package com.mycompany.app.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.mycompany.app.model.openai.CompletionRequest;
import com.mycompany.app.model.openai.ChatCompletionRequest;
import com.mycompany.app.model.openai.CompletionResponse;
import com.mycompany.app.model.openai.ChatCompletionResponse;
import com.mycompany.app.service.AiService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;

@RestController
@RequestMapping("/v1")
@Tag(name = "OpenAI-compatible API", description = "OpenAI-compatible API for interacting with AI models backed by Vertex AI.")
public class AiController {

    private final AiService aiService;

    public AiController(AiService aiService) {
        this.aiService = aiService;
    }


    @PostMapping("/chat/completions")
    @Operation(summary = "Creates a model response for the given chat conversation.", description = "Creates a model response for the given chat conversation, using a Gemma model on Vertex AI. This endpoint is compatible with the OpenAI Chat Completions API.", responses = {
            @ApiResponse(responseCode = "200", description = "Successful response from the AI model.", content = @Content(mediaType = "application/json", schema = @Schema(implementation = ChatCompletionResponse.class))),
            @ApiResponse(responseCode = "400", description = "Invalid request format."),
            @ApiResponse(responseCode = "500", description = "Internal server error.") })
    public ChatCompletionResponse createChatCompletion(@Valid @RequestBody ChatCompletionRequest request) {
        ChatCompletionResponse response = aiService.createChatCompletion(request);
        return response;
    }

    @PostMapping("/completions")
    @Operation(summary = "Creates a completion for the provided prompt.", description = "Creates a completion for the provided prompt, using a Gemma model on Vertex AI. This endpoint is compatible with the OpenAI Completions API.", responses = {
            @ApiResponse(responseCode = "200", description = "Successful response from the AI model.", content = @Content(mediaType = "application/json", schema = @Schema(implementation = CompletionResponse.class))),
            @ApiResponse(responseCode = "400", description = "Invalid request format."),
            @ApiResponse(responseCode = "500", description = "Internal server error.") })
    public CompletionResponse createCompletion(@Valid @RequestBody CompletionRequest request) {
        CompletionResponse response = aiService.createCompletion(request);
        return response;
    }
}