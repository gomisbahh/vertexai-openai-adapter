# Stage 1: Build the application with Maven
# This stage uses a full JDK and Maven environment to build the .jar file.
FROM maven:3.8.5-openjdk-17 AS builder
WORKDIR /app

# Copy the pom.xml and download dependencies first to leverage Docker's layer cache
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy the rest of the source code
COPY src ./src

# Package the application
RUN mvn package -DskipTests
# RUN mvn package


# Stage 2: Create the final, lean production image
# This stage uses a slim JRE, which is much smaller than the build environment.
FROM openjdk:17-jdk-slim
WORKDIR /app

# Copy only the built application .jar file from the 'builder' stage
COPY --from=builder /app/target/*.jar app.jar

# Expose the port the application runs on
EXPOSE 8080

# Command to run the application when the container starts
CMD ["java", "-jar", "app.jar"]