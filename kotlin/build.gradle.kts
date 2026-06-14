plugins {
    kotlin("jvm") version "2.1.21"
}

group = "com.dho"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
}

kotlin {
    jvmToolchain(17)
}
