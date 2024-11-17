using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections;
using UnityEngine.Networking;
using System.Collections.Generic;
using System.Linq;

public class BodyPartSelector : MonoBehaviour
{
    public Button chestButton;
    public Button shoulderButton;
    public Button abdomenButton;
    public Button backButton;
    public Button armButton;
    public Button thighButton;
    public Button ankleButton;
    public Button waistButton;
    public Button calfButton;
    public Button doneButton;
    public TMP_Text resultText;  // TMP_Text�� ����

    private string serverUrl = "http://127.0.0.1:5000/get_exercises";
    private List<string> selectedBodyParts = new List<string>();

    void Start()
    {
        if (chestButton == null || resultText == null || doneButton == null)
        {
            Debug.LogError("Button or resultText is not assigned in the inspector.");
            return;
        }

        chestButton.onClick.AddListener(() => OnBodyPartClicked("chest", chestButton));
        shoulderButton.onClick.AddListener(() => OnBodyPartClicked("shoulders", shoulderButton));
        abdomenButton.onClick.AddListener(() => OnBodyPartClicked("abs", abdomenButton));
        backButton.onClick.AddListener(() => OnBodyPartClicked("back", backButton));
        armButton.onClick.AddListener(() => OnBodyPartClicked("arms", armButton));
        thighButton.onClick.AddListener(() => OnBodyPartClicked("thighs", thighButton));
        ankleButton.onClick.AddListener(() => OnBodyPartClicked("ankles", ankleButton));
        waistButton.onClick.AddListener(() => OnBodyPartClicked("waist", waistButton));
        calfButton.onClick.AddListener(() => OnBodyPartClicked("calves", calfButton));

        doneButton.onClick.AddListener(OnDoneButtonClicked);
    }

    void OnBodyPartClicked(string bodyPart, Button clickedButton)
    {
        if (selectedBodyParts.Contains(bodyPart))
        {
            selectedBodyParts.Remove(bodyPart);
            clickedButton.GetComponent<Image>().color = Color.white;
        }
        else
        {
            selectedBodyParts.Add(bodyPart);
            clickedButton.GetComponent<Image>().color = Color.green;
        }
    }

    void OnDoneButtonClicked()
    {
        if (selectedBodyParts.Count > 0)
        {
            string bodyPartsString = string.Join(",", selectedBodyParts);
            SendRequest(bodyPartsString);
        }
        else
        {
            resultText.text = "Please select at least one body part!";
        }
    }

    void SendRequest(string bodyParts)
    {
        StartCoroutine(SendRequestCoroutine(bodyParts));
    }

    IEnumerator SendRequestCoroutine(string bodyParts)
    {
        // JSON �����͸� ���Ŀ� �°� ����
        string jsonData = "{\"body_parts\":[" + string.Join(",", selectedBodyParts.Select(x => $"\"{x}\"")) + "]}";

        using (UnityWebRequest request = UnityWebRequest.Put(serverUrl, jsonData))
        {
            request.method = UnityWebRequest.kHttpVerbPOST;
            request.SetRequestHeader("Content-Type", "application/json");
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success) // ��û�� �����ϸ�
            {
                // ���� ������ �޾Ƽ� ���
                Debug.Log("Request Success. Response: " + request.downloadHandler.text);

                // ���� ���� �ؽ�Ʈ�� �޾Ƽ� ó��
                string responseText = request.downloadHandler.text;
                ProcessResponse(responseText); // ���� �ؽ�Ʈ�� ó��
            }
            else
            {
                // ��û ���� �� ���� ���
                Debug.LogError("Request failed. Error: " + request.error);
                resultText.text = "Error: " + request.error; // ���� �޽��� ǥ��
            }
        }
    }


    void ProcessResponse(string responseText)
    {
        var jsonResponse = JsonUtility.FromJson<ExerciseResponse>(responseText);
        resultText.text = "Recommended exercises:\n";

        foreach (string exercise in jsonResponse.exercises)
        {
            resultText.text += exercise + "\n";
        }

        Debug.Log("Updated ResultText: " + resultText.text);  // �߰��� �α�
    }


    [System.Serializable]
    public class ExerciseResponse
    {
        public string[] exercises;
    }
}

