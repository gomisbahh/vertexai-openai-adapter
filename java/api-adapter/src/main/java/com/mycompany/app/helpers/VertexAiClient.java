package com.mycompany.app.helpers;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;


public class VertexAiClient {

  private final String ENDPOINT_URL;
  private final HttpClient httpClient;
  private final Gson gson;
  private final GoogleCredentials credentials;


  public VertexAiClient(String projectId, String location, String endpointId) {

     String endpoint_type = getEnv("ENDPOINT_TYPE", "PUBLIC");
     String endpoint_ip = getEnv("ENDPOINT_IP", "10.132.8.10");
     String endpoint_protocol = getEnv("ENDPOINT_PROTOCOL", "https");
     switch (endpoint_type) {
         case "PUBLIC":
          this.ENDPOINT_URL = String.format(
              "https://%s.%s-%s.prediction.vertexai.goog/v1/projects/%s/locations/%s/endpoints/%s:predict",
              endpointId, location, projectId, projectId, location, endpointId);
             break;
          case "PRIVATE":
          this.ENDPOINT_URL = String.format(
              "%s://%s/v1/projects/%s/locations/%s/endpoints/%s:predict",
              endpoint_protocol, endpoint_ip, projectId, location, endpointId);
             break;
         default:
            this.ENDPOINT_URL = String.format(
            "https://%s.%s-%s.prediction.vertexai.goog/v1/projects/%s/locations/%s/endpoints/%s:predict",
            endpointId, location, projectId, projectId, location, endpointId);
     }

     System.out.println("Endpoint URL: " + this.ENDPOINT_URL + "\n");

    
    HttpClient.Builder clientBuilder = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(60));

    this.httpClient = clientBuilder.build(); 
    this.gson = new Gson();

    try {
      // Initialize Google Cloud credentials using Application Default Credentials (ADC).
      // ADC will automatically find credentials on GKE (via Workload Identity),
      // or from `gcloud auth application-default login` for local development.
      this.credentials = GoogleCredentials.getApplicationDefault()
              .createScoped("https://www.googleapis.com/auth/cloud-platform");
    } catch (IOException e) {
      throw new RuntimeException("Could not get application default credentials.", e);
    }
  }

  /**
   * Creates the request payload for Gemma3 model
   */
  private String createRequestPayload(String prompt) {

    JsonObject messageObject = new JsonObject();
    messageObject.addProperty("role", "user");
    messageObject.addProperty("content", prompt);

    JsonArray messagesArray = new JsonArray();
    messagesArray.add(messageObject);

    JsonObject instanceObject = new JsonObject();
    instanceObject.addProperty("@requestFormat", "chatCompletions");
    instanceObject.add("messages", messagesArray);
    instanceObject.addProperty("max_tokens", 100);

    JsonObject payload = new JsonObject();
    payload.add("instances", gson.toJsonTree(new JsonObject[]{instanceObject}));


    return gson.toJson(payload);
  }

  /**
   * Sends request to Vertex AI endpoint and returns the summary
   */
  public String prompt(String text) throws IOException, InterruptedException {
    // Refresh credentials if needed
    credentials.refreshIfExpired();
    String accessToken = credentials.getAccessToken().getTokenValue();
    String requestBody = createRequestPayload(text);

    HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(ENDPOINT_URL))
            .header("Authorization", "Bearer " + accessToken.trim())
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .timeout(Duration.ofSeconds(60))
            .build();

    HttpResponse<String> response = httpClient.send(request,
            HttpResponse.BodyHandlers.ofString());

    if (response.statusCode() != 200) {
      throw new RuntimeException("Request failed with status: " + response.statusCode() +
              "\nResponse: " + response.body());
    }

    return parseResponse(response.body());
  }

  /**
   * Extracts the summary text from the API response
   */
  private String parseResponse(String responseBody) {
    try {
      JsonObject rootObject = JsonParser.parseString(responseBody).getAsJsonObject();

      JsonObject predictionsObject = rootObject.getAsJsonObject("predictions");
      JsonArray choicesArray = predictionsObject.getAsJsonArray("choices");
      JsonElement firstChoiceObject = choicesArray.getAsJsonArray().get(0);
      JsonObject messageObject = firstChoiceObject.getAsJsonObject().getAsJsonObject("message");
      return messageObject.getAsJsonPrimitive("content").getAsString();

    } catch (Exception e) {
      e.printStackTrace();
      return "Error parsing response: " + e.getMessage() + "\nFull response: " + responseBody;

    }
  }

  /**
   * Helper method to get an environment variable or return a default value.
   */
  private static String getEnv(String name, String defaultValue) {
    String value = System.getenv(name);
    if (value == null || value.isEmpty()) {
      return defaultValue;
    }
    return value;
  }

  }