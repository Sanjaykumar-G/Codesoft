import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class rs {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Predefined set of movies and their genres
        Map<String, String> movies = new HashMap<>();
        movies.put("The Shawshank Redemption", "Drama");
        movies.put("The Godfather", "Crime");
        movies.put("The Dark Knight", "Action");
        movies.put("Pulp Fiction", "Crime");
        movies.put("Forrest Gump", "Drama");
        movies.put("The Matrix", "Action");

        // Get user preferences
        System.out.println("Welcome to the Movie Recommendation System!");
        System.out.println("Please enter your favorite movie genre:");
        String userGenrePreference = scanner.nextLine();

        // Recommend a movie based on user preferences
        String recommendedMovie = recommendMovie(movies, userGenrePreference);
        System.out.println("I recommend you watch: " + recommendedMovie);

        scanner.close();
    }

    // Recommend a movie based on user's genre preference
    public static String recommendMovie(Map<String, String> movies, String userPreference) {
        // Convert user preference to lowercase for case-insensitivity
        userPreference = userPreference.toLowerCase();

        // Iterate through movies to find a match with the user's preference
        for (Map.Entry<String, String> entry : movies.entrySet()) {
            if (entry.getValue().toLowerCase().equals(userPreference)) {
                return entry.getKey();
            }
        }

        // If no match is found, provide a generic recommendation
        return "Unfortunately, we couldn't find a specific movie in that genre. How about trying a different genre?";
    }
}
