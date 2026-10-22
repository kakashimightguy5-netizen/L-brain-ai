# Verified frontend parameter count

{
  "latest_frontend_selectors": [
    {
      "name": "domain selector v2",
      "coefficient_shape": [
        5,
        3593
      ],
      "weights": 17965,
      "biases": 5,
      "parameters": 17970,
      "tfidf_idf_values": 3593
    },
    {
      "name": "technology selector v2",
      "coefficient_shape": [
        5,
        392
      ],
      "weights": 1960,
      "biases": 5,
      "parameters": 1965,
      "tfidf_idf_values": 392
    }
  ],
  "combined_classifier_parameters": 19935,
  "additional_tfidf_idf_values": 3985,
  "knowledge_records": 30,
  "scope": "Latest two frontend selector versions; stored code, archived versions and earlier SNN/CNN models excluded. Not one integrated generative model."
}

Counts are classifier coefficients plus biases. TF-IDF IDF values are separately reported fitted preprocessing statistics. Stored source/knowledge records are not neural parameters. Earlier vision/SNN experts belong to separate experiments.
