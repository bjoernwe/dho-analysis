plugins {
    kotlin("jvm") version "2.4.10"
    id("org.jetbrains.kotlin.plugin.dataframe") version "2.4.10"
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
    implementation("org.jetbrains.kotlinx:dataframe:1.0.0-dev-10964")
    implementation("com.github.haifengl:smile-core:3.1.1")
    implementation("org.xerial:sqlite-jdbc:3.46.1.3")
    implementation("org.apache.opennlp:opennlp-tools:2.5.9")
    implementation("me.tongfei:progressbar:0.10.1")
}

// Add test dependencies
dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}

tasks.test {
    useJUnitPlatform()
}

application {
    mainClass.set("experiments.SentencePcaKt")
}

kotlin {
    jvmToolchain(17)
}
