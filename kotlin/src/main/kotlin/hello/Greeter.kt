package hello

fun greet(name: String): String = "Hello, $name!"

class Greeter(private val greeting: String = "Hello") {
    fun greet(name: String): String = "$greeting, $name!"
}
