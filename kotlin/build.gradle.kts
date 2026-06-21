plugins {
    kotlin("jvm") version "2.2.21"
    id("org.jetbrains.kotlin.plugin.dataframe") version "2.2.21"
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
    implementation("org.jetbrains.kotlinx:dataframe:1.0.0-Beta5")
    implementation("org.xerial:sqlite-jdbc:3.46.1.3")
}

application {
    mainClass.set("examples.ClassificationKt")
}

kotlin {
    jvmToolchain(17)
}
