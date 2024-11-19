using MySql.Data.MySqlClient;
using System.Collections.Generic;
using UnityEngine;

public class DatabaseManager : MonoBehaviour
{
    private static DatabaseManager _instance;
    public static DatabaseManager Instance
    {
        get
        {
            if (_instance == null)
            {
                GameObject go = new GameObject("DatabaseManager");
                _instance = go.AddComponent<DatabaseManager>();
                DontDestroyOnLoad(go);
            }
            return _instance;
        }
    }

    private string connectionString = "Server=YOUR_SERVER_IP;Database=YOUR_DATABASE;User ID=YOUR_USERNAME;Password=YOUR_PASSWORD;";

    // 쿼리 실행
    public void ExecuteQuery(List<string> selectedValues)
    {
        if (selectedValues.Count == 0)
        {
            Debug.LogWarning("No values selected!");
            return;
        }

        string query = "SELECT * FROM ExerciseData WHERE target_muscle IN (";

        // 선택된 값들을 쿼리로 추가
        for (int i = 0; i < selectedValues.Count; i++)
        {
            query += $"'{selectedValues[i]}'";
            if (i < selectedValues.Count - 1)
            {
                query += ",";
            }
        }
        query += ");";

        Debug.Log($"Executing query: {query}");

        try
        {
            using (MySqlConnection connection = new MySqlConnection(connectionString))
            {
                connection.Open();
                using (MySqlCommand command = new MySqlCommand(query, connection))
                {
                    using (MySqlDataReader reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            string exerciseName = reader["exercise_name"].ToString();
                            Debug.Log($"Exercise: {exerciseName}");
                        }
                    }
                }
            }
        }
        catch (MySqlException ex)
        {
            Debug.LogError($"Database error: {ex.Message}");
        }
    }
}
