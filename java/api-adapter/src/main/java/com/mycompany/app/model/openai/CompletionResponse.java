package com.mycompany.app.model.openai;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

public class CompletionResponse {
    private String id;
    private String object;
    private long created;
    private String model;
    private List<Choice> choices;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getObject() { return object; }
    public void setObject(String object) { this.object = object; }
    public long getCreated() { return created; }
    public void setCreated(long created) { this.created = created; }
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
    public List<Choice> getChoices() { return choices; }
    public void setChoices(List<Choice> choices) { this.choices = choices; }

    public static class Choice {
        private String text;
        private int index;
        @JsonProperty("finish_reason")
        private String finishReason;

        public String getText() { return text; }
        public void setText(String text) { this.text = text; }
        public int getIndex() { return index; }
        public void setIndex(int index) { this.index = index; }
        public String getFinishReason() { return finishReason; }
        public void setFinishReason(String finishReason) { this.finishReason = finishReason; }
    }
}