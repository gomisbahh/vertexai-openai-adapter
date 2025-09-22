package com.mycompany.app.model.openai;

import jakarta.validation.constraints.NotBlank;

public class CompletionRequest {
    private String model;

    @NotBlank
    private String prompt;

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getPrompt() {
        return prompt;
    }

    public void setPrompt(String prompt) {
        this.prompt = prompt;
    }
}