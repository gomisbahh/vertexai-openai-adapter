package org.springframework.ai.openai.samples.helloworld;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    @Bean
    public CommandLineRunner run(ChatClient.Builder chatClientBuilder) {
        return args -> {
            String prompt = "Tell me a fun fact about giraffes.";
            // The ChatClient will automatically use the configuration from application.properties
            String response = chatClientBuilder.build().prompt()
                    .user(prompt)
                    .call()
                    .content();

            System.out.println("----------------------------------------");
            System.out.println("Response from AI:");
            System.out.println(response);
            System.out.println("----------------------------------------");
        };
    }
}