plugins {
    kotlin("jvm") version "2.1.21"
    application
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "com.dho"
version = "0.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("ai.djl.huggingface:tokenizers:0.33.0")
    implementation("com.microsoft.onnxruntime:onnxruntime_gpu:1.22.0")
    implementation("org.jetbrains.kotlinx:dataframe:0.15.0")
}

application {
    mainClass.set("MainKt")
}

kotlin {
    jvmToolchain(17)
}
