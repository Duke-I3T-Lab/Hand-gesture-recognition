using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.IO;
using System.Text;
using System.Threading.Tasks;
using System;

#if WINDOWS_UWP
using Windows.Storage;
#endif

using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using TMPro;
using Microsoft.MixedReality.Toolkit;
using System.Threading;
using System.Diagnostics;
using UnityEngine.Networking;

// This script establishes a TCP connection for labeling collected hand gesture datasets
// to be used for ML training purposes.

public class TCP_TrackingLogger : MonoBehaviour
{
    #region Constants to modify
    private const string DataSuffix = "TCPTracking";
    private const string CSVHeader = "Label," + "Time," + "Counter," 
                                     + "IndexDistalJoint," + "IndexKnuckle," + "IndexMetacarpal," + "IndexMiddleJoint," + "IndexTip,"
                                     + "MiddleDistalJoint," + "MiddleKnuckle," + "MiddleMetacarpal," + "MiddleMiddleJoint," + "MiddleTip," + "Palm,"
                                     + "PinkyDistalJoint," + "PinkyKnuckle," + "PinkyMetacarpal," + "PinkyMiddleJoint," + "PinkyTip,"
                                     + "RingDistalJoint," + "RingKnuckle," + "RingMetacarpal," + "RingMiddleJoint," + "RingTip,"
                                     + "ThumbDistalJoint," + "ThumbMetacarpalJoint," + "ThumbProximalJoint," + "ThumbTip," + "Wrist,"
                                     + "IndexDistalJoint," + "IndexKnuckle," + "IndexMetacarpal," + "IndexMiddleJoint," + "IndexTip,"
                                     + "MiddleDistalJoint," + "MiddleKnuckle," + "MiddleMetacarpal," + "MiddleMiddleJoint," + "MiddleTip," + "Palm,"
                                     + "PinkyDistalJoint," + "PinkyKnuckle," + "PinkyMetacarpal," + "PinkyMiddleJoint," + "PinkyTip,"
                                     + "RingDistalJoint," + "RingKnuckle," + "RingMetacarpal," + "RingMiddleJoint," + "RingTip,"
                                     + "ThumbDistalJoint," + "ThumbMetacarpalJoint," + "ThumbProximalJoint," + "ThumbTip," + "Wrist";
    private const string SessionFolderRoot = "CSVLogger";
    #endregion

    #region private members
    private string m_sessionPath;
    private string m_filePath;
    private string m_recordingId;
    private string m_sessionId;

    private StringBuilder m_csvData;
    #endregion
    #region public members
    public string RecordingInstance => m_recordingId;
    #endregion

    private int csv_started, data_counter;
    private int hand_logging = 1;
    public TMP_Text logger_text;

    Stopwatch LoggingstopWatch = new Stopwatch();

    // Change hostname to the server (labeler) computer's IPv4, make sure HoloLens is on the same WiFi as server
    private string hostname = "http://192.168.1.13:8000";         
    private string data_received = "";
    private string loggerData = "start";

    // Start is called before the first frame update
    async void Start()
    {
        csv_started = 0;
        data_counter = 0;
        LoggingstopWatch.Start();

        await MakeNewSession();
    }

    void Update()
    {
        if (csv_started == 0)
        {
      
            StartNewCSV();
            csv_started = 1;
            UnityEngine.Debug.Log("New CSV started");
        }
        else if (csv_started == 1) 
        {
            StartCoroutine("logging_tracking");
        }

        print("LoggerData: " + loggerData);
        loggerData = "";            // reset the logger data being sent to Python server

        StartCoroutine(Receive(hostname));
    }

    public void logging_tracking()
    {
        List<String> rowData = new List<String>();
        // add the current time and frame number (first 2 rows)
        rowData.Add(data_received);
        rowData.Add(DateTime.Now.ToString("HH:mm:ss.fff"));  // in csv column "Time"
        rowData.Add((data_counter).ToString());

        bool? calibrationStatus = CoreServices.InputSystem?.EyeGazeProvider?.IsEyeCalibrationValid;

        if (calibrationStatus != null)
        {
            if (hand_logging == 1)
            {
                // store hand tracking data 
                add_rightHand_data(rowData);
                add_leftHand_data(rowData);
            }

            // add each column data to one string (for sending it to Python server)
            foreach (string str in rowData)
            {
                loggerData += str.ToString() + ", ";
            }

            print("LoggerData: " + loggerData);

            if (csv_started == 1)
            {
                AddRow(rowData);    // add all data to CSV
                FlushData();        // flush
            }

            data_counter++;
        }
        else
        {
            //logger_text.text = string.Format("Need Eye Calibration");
        }
    }

  

    public void add_rightHand_data(List<String> rowData)
    {
        MixedRealityPose pose;
        string index_disj, index_knuckle, index_mc, index_middlej, index_tip;
        string middle_disj, middle_knuckle, middle_mc, middle_middlej, middle_tip;
        string pinky_disj, pinky_knuckle, pinky_mc, pinky_middlej, pinky_tip;
        string ring_disj, ring_knuckle, ring_mc, ring_middlej, ring_tip;
        string thumb_disj, thumb_mcj, thumb_proxj, thumb_tip;
        string palm, wrist;

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexDistalJoint, Handedness.Right, out pose))
        {
            index_disj = pose.Position.ToString("F3");
            index_disj = index_disj.Replace(",", "/");
            rowData.Add(index_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexKnuckle, Handedness.Right, out pose))
        {
            index_knuckle = pose.Position.ToString("F3");
            index_knuckle = index_knuckle.Replace(",", "/");
            rowData.Add(index_knuckle);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexMetacarpal, Handedness.Right, out pose))
        {
            index_mc = pose.Position.ToString("F3");
            index_mc = index_mc.Replace(",", "/");
            rowData.Add(index_mc);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexMiddleJoint, Handedness.Right, out pose))
        {
            index_middlej = pose.Position.ToString("F3");
            index_middlej = index_middlej.Replace(",", "/");
            rowData.Add(index_middlej);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexTip, Handedness.Right, out pose))
        {
            index_tip = pose.Position.ToString("F3");
            index_tip = index_tip.Replace(",", "/");
            rowData.Add(index_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleDistalJoint, Handedness.Right, out pose))
        {
            middle_disj = pose.Position.ToString("F3");
            middle_disj = middle_disj.Replace(",", "/");
            rowData.Add(middle_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleKnuckle, Handedness.Right, out pose))
        {
            middle_knuckle = pose.Position.ToString("F3");
            middle_knuckle = middle_knuckle.Replace(",", "/");
            rowData.Add(middle_knuckle);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleMetacarpal, Handedness.Right, out pose))
        {
            middle_mc = pose.Position.ToString("F3");
            middle_mc = middle_mc.Replace(",", "/");
            rowData.Add(middle_mc);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleMiddleJoint, Handedness.Right, out pose))
        {
            middle_middlej = pose.Position.ToString("F3");
            middle_middlej = middle_middlej.Replace(",", "/");
            rowData.Add(middle_middlej);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleTip, Handedness.Right, out pose))
        {
            middle_tip = pose.Position.ToString("F3");
            middle_tip = middle_tip.Replace(",", "/");
            rowData.Add(middle_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.Palm, Handedness.Right, out pose))
        {
            palm = pose.Position.ToString("F3");
            palm = palm.Replace(",", "/");
            rowData.Add(palm);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyDistalJoint, Handedness.Right, out pose))
        {
            pinky_disj = pose.Position.ToString("F3");
            pinky_disj = pinky_disj.Replace(",", "/");
            rowData.Add(pinky_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyKnuckle, Handedness.Right, out pose))
        {
            pinky_knuckle = pose.Position.ToString("F3");
            pinky_knuckle = pinky_knuckle.Replace(",", "/");
            rowData.Add(pinky_knuckle);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyMetacarpal, Handedness.Right, out pose))
        {
            pinky_mc = pose.Position.ToString("F3");
            pinky_mc = pinky_mc.Replace(",", "/");
            rowData.Add(pinky_mc);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyMiddleJoint, Handedness.Right, out pose))
        {
            pinky_middlej = pose.Position.ToString("F3");
            pinky_middlej = pinky_middlej.Replace(",", "/");
            rowData.Add(pinky_middlej);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyTip, Handedness.Right, out pose))
        {
            pinky_tip = pose.Position.ToString("F3");
            pinky_tip = pinky_tip.Replace(",", "/");
            rowData.Add(pinky_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingDistalJoint, Handedness.Right, out pose))
        {
            ring_disj = pose.Position.ToString("F3");
            ring_disj = ring_disj.Replace(",", "/");
            rowData.Add(ring_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingKnuckle, Handedness.Right, out pose))
        {
            ring_knuckle = pose.Position.ToString("F3");
            ring_knuckle = ring_knuckle.Replace(",", "/");
            rowData.Add(ring_knuckle);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingMetacarpal, Handedness.Right, out pose))
        {
            ring_mc = pose.Position.ToString("F3");
            ring_mc = ring_mc.Replace(",", "/");
            rowData.Add(ring_mc);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingMiddleJoint, Handedness.Right, out pose))
        {
            ring_middlej = pose.Position.ToString("F3");
            ring_middlej = ring_middlej.Replace(",", "/");
            rowData.Add(ring_middlej);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingTip, Handedness.Right, out pose))
        {
            ring_tip = pose.Position.ToString("F3");
            ring_tip = ring_tip.Replace(",", "/");
            rowData.Add(ring_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbDistalJoint, Handedness.Right, out pose))
        {
            thumb_disj = pose.Position.ToString("F3");
            thumb_disj = thumb_disj.Replace(",", "/");
            rowData.Add(thumb_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbMetacarpalJoint, Handedness.Right, out pose))
        {
            thumb_mcj = pose.Position.ToString("F3");
            thumb_mcj = thumb_mcj.Replace(",", "/");
            rowData.Add(thumb_mcj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbProximalJoint, Handedness.Right, out pose))
        {
            thumb_proxj = pose.Position.ToString("F3");
            thumb_proxj = thumb_proxj.Replace(",", "/");
            rowData.Add(thumb_proxj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbTip, Handedness.Right, out pose))
        {
            thumb_tip = pose.Position.ToString("F3");
            thumb_tip = thumb_tip.Replace(",", "/");
            rowData.Add(thumb_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.Wrist, Handedness.Right, out pose))
        {
            wrist = pose.Position.ToString("F3");
            wrist = wrist.Replace(",", "/");
            rowData.Add(wrist);
        }
        else
            rowData.Add("0");
    }

    public void add_leftHand_data(List<String> rowData)
    {
        MixedRealityPose pose;
        string index_disj, index_knuckle, index_mc, index_middlej, index_tip;
        string middle_disj, middle_knuckle, middle_mc, middle_middlej, middle_tip;
        string pinky_disj, pinky_knuckle, pinky_mc, pinky_middlej, pinky_tip;
        string ring_disj, ring_knuckle, ring_mc, ring_middlej, ring_tip;
        string thumb_disj, thumb_mcj, thumb_proxj, thumb_tip;
        string palm, wrist;

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexDistalJoint, Handedness.Left, out pose))
        {
            index_disj = pose.Position.ToString("F3");
            index_disj = index_disj.Replace(",", "/");
            rowData.Add(index_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexKnuckle, Handedness.Left, out pose))
        {
            index_knuckle = pose.Position.ToString("F3");
            index_knuckle = index_knuckle.Replace(",", "/");
            rowData.Add(index_knuckle);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexMetacarpal, Handedness.Left, out pose))
        {
            index_mc = pose.Position.ToString("F3");
            index_mc = index_mc.Replace(",", "/");
            rowData.Add(index_mc);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexMiddleJoint, Handedness.Left, out pose))
        {
            index_middlej = pose.Position.ToString("F3");
            index_middlej = index_middlej.Replace(",", "/");
            rowData.Add(index_middlej);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexTip, Handedness.Left, out pose))
        {
            index_tip = pose.Position.ToString("F3");
            index_tip = index_tip.Replace(",", "/");
            rowData.Add(index_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleDistalJoint, Handedness.Left, out pose))
        {
            middle_disj = pose.Position.ToString("F3");
            middle_disj = middle_disj.Replace(",", "/");
            rowData.Add(middle_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleKnuckle, Handedness.Left, out pose))
        {
            middle_knuckle = pose.Position.ToString("F3");
            middle_knuckle = middle_knuckle.Replace(",", "/");
            rowData.Add(middle_knuckle);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleMetacarpal, Handedness.Left, out pose))
        {
            middle_mc = pose.Position.ToString("F3");
            middle_mc = middle_mc.Replace(",", "/");
            rowData.Add(middle_mc);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleMiddleJoint, Handedness.Left, out pose))
        {
            middle_middlej = pose.Position.ToString("F3");
            middle_middlej = middle_middlej.Replace(",", "/");
            rowData.Add(middle_middlej);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.MiddleTip, Handedness.Left, out pose))
        {
            middle_tip = pose.Position.ToString("F3");
            middle_tip = middle_tip.Replace(",", "/");
            rowData.Add(middle_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.Palm, Handedness.Left, out pose))
        {
            palm = pose.Position.ToString("F3");
            palm = palm.Replace(",", "/");
            rowData.Add(palm);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyDistalJoint, Handedness.Left, out pose))
        {
            pinky_disj = pose.Position.ToString("F3");
            pinky_disj = pinky_disj.Replace(",", "/");
            rowData.Add(pinky_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyKnuckle, Handedness.Left, out pose))
        {
            pinky_knuckle = pose.Position.ToString("F3");
            pinky_knuckle = pinky_knuckle.Replace(",", "/");
            rowData.Add(pinky_knuckle);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyMetacarpal, Handedness.Left, out pose))
        {
            pinky_mc = pose.Position.ToString("F3");
            pinky_mc = pinky_mc.Replace(",", "/");
            rowData.Add(pinky_mc);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyMiddleJoint, Handedness.Left, out pose))
        {
            pinky_middlej = pose.Position.ToString("F3");
            pinky_middlej = pinky_middlej.Replace(",", "/");
            rowData.Add(pinky_middlej);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.PinkyTip, Handedness.Left, out pose))
        {
            pinky_tip = pose.Position.ToString("F3");
            pinky_tip = pinky_tip.Replace(",", "/");
            rowData.Add(pinky_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingDistalJoint, Handedness.Left, out pose))
        {
            ring_disj = pose.Position.ToString("F3");
            ring_disj = ring_disj.Replace(",", "/");
            rowData.Add(ring_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingKnuckle, Handedness.Left, out pose))
        {
            ring_knuckle = pose.Position.ToString("F3");
            ring_knuckle = ring_knuckle.Replace(",", "/");
            rowData.Add(ring_knuckle);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingMetacarpal, Handedness.Left, out pose))
        {
            ring_mc = pose.Position.ToString("F3");
            ring_mc = ring_mc.Replace(",", "/");
            rowData.Add(ring_mc);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingMiddleJoint, Handedness.Left, out pose))
        {
            ring_middlej = pose.Position.ToString("F3");
            ring_middlej = ring_middlej.Replace(",", "/");
            rowData.Add(ring_middlej);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.RingTip, Handedness.Left, out pose))
        {
            ring_tip = pose.Position.ToString("F3");
            ring_tip = ring_tip.Replace(",", "/");
            rowData.Add(ring_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbDistalJoint, Handedness.Left, out pose))
        {
            thumb_disj = pose.Position.ToString("F3");
            thumb_disj = thumb_disj.Replace(",", "/");
            rowData.Add(thumb_disj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbMetacarpalJoint, Handedness.Left, out pose))
        {
            thumb_mcj = pose.Position.ToString("F3");
            thumb_mcj = thumb_mcj.Replace(",", "/");
            rowData.Add(thumb_mcj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbProximalJoint, Handedness.Left, out pose))
        {
            thumb_proxj = pose.Position.ToString("F3");
            thumb_proxj = thumb_proxj.Replace(",", "/");
            rowData.Add(thumb_proxj);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbTip, Handedness.Left, out pose))
        {
            thumb_tip = pose.Position.ToString("F3");
            thumb_tip = thumb_tip.Replace(",", "/");
            rowData.Add(thumb_tip);
        }
        else
            rowData.Add("0");

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.Wrist, Handedness.Left, out pose))
        {
            wrist = pose.Position.ToString("F3");
            wrist = wrist.Replace(",", "/");
            rowData.Add(wrist);
        }
        else
            rowData.Add("0");
    }

    public string string_replace(string data)      // removing parentheses from input
    {
        data = data.Replace("(", "");
        data = data.Replace(")", "");
        return data;
    }

    public async Task MakeNewSession()
    {
        m_sessionId = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string rootPath = "";
#if WINDOWS_UWP
        StorageFolder sessionParentFolder = await KnownFolders.PicturesLibrary
            .CreateFolderAsync(SessionFolderRoot,
            CreationCollisionOption.OpenIfExists);
        rootPath = sessionParentFolder.Path;
#else
        rootPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), SessionFolderRoot);
        if (!Directory.Exists(rootPath)) Directory.CreateDirectory(rootPath);
#endif
        m_sessionPath = Path.Combine(rootPath, m_sessionId);
        Directory.CreateDirectory(m_sessionPath);
        UnityEngine.Debug.Log("CSVLogger logging data to " + m_sessionPath);
    }

    public void StartNewCSV()
    {
        m_recordingId = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
        var filename = m_recordingId + "-" + DataSuffix + ".csv";
        m_filePath = Path.Combine(Application.persistentDataPath, filename);
        if (m_csvData != null)
        {
            EndCSV();
        }
        m_csvData = new StringBuilder();
        m_csvData.AppendLine(",,,Right Hand,,,,,,,,,,,,,,,,,,,,,,,,,,Left Hand");
        m_csvData.AppendLine(CSVHeader);
        logger_text.text = string.Format("CSV Started");
    }

    public void EndCSV()
    {
        if (m_csvData == null)
        {
            return;
        }
        using (var csvWriter = new StreamWriter(m_filePath, true))
        {
            csvWriter.Write(m_csvData.ToString());
        }
        m_recordingId = null;
        m_csvData = null;
    }

    public void OnDestroy()
    {
        EndCSV();
    }

    public void AddRow(List<String> rowData)
    {
        AddRow(string.Join(",", rowData.ToArray()));
    }

    public void AddRow(string row)
    {
        m_csvData.AppendLine(row);
    }

    /// <summary>
    /// Writes all current data to current file
    /// </summary>
    public void FlushData()
    {
        using (var csvWriter = new StreamWriter(m_filePath, true))
        {
            csvWriter.Write(m_csvData.ToString());
        }
        m_csvData.Clear();
    }

    /// <summary>
    /// Returns a row populated with common start data like
    /// recording id, session id, timestamp
    /// </summary>
    /// <returns></returns>
    public List<String> RowWithStartData()
    {
        List<String> rowData = new List<String>();
        rowData.Add(Time.timeSinceLevelLoad.ToString("##.000"));
        rowData.Add(m_recordingId);
        rowData.Add(m_recordingId);
        return rowData;
    }

    IEnumerator Receive(string url)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(url))
        {
            // Request and wait for the desired page.
            yield return webRequest.SendWebRequest();

            string[] pages = url.Split('/');
            int page = pages.Length - 1;

            if (webRequest.isNetworkError || webRequest.isHttpError)
            {
                //UnityEngine.Debug.Log(webRequest.error);
            }
            else
            {
                UnityEngine.Debug.Log(pages[page] + ":\nReceived: " + webRequest.downloadHandler.text);
                data_received = webRequest.downloadHandler.text;
                //UnityEngine.Debug.Log(data_received.Length);
                if (data_received.Length < 1)
                {
                    //UnityEngine.Debug.Log("No Data Received");
                }
                else
                {
                    UnityEngine.Debug.Log("Received: " + data_received);
                    switch (data_received)
                    {
                        case "0":
                            logger_text.text = string.Format("0: Miscellaneous");
                            break;
                        case "`":
                            logger_text.text = string.Format("tilde: LH Blossom");
                            break;
                        case "1":
                            logger_text.text = string.Format("1: LH Grab");
                            break;
                        case "2":
                            logger_text.text = string.Format("2: LH Swipe Right");
                            break;
                        case "3":
                            logger_text.text = string.Format("3: LH Swipe Left");
                            break;
                        case "4":
                            logger_text.text = string.Format("4: RH Blossom");
                            break;
                        case "5":
                            logger_text.text = string.Format("5: RH Grab");
                            break;
                        case "6":
                            logger_text.text = string.Format("6: RH Swipe Right");
                            break;
                        case "7":
                            logger_text.text = string.Format("7: RH Swipe Left");
                            break;
                        default:
                            logger_text.text = string.Format("Invalid inputs");
                            break;

                    }
                }
            }
        }
    }

    IEnumerator Upload(string url, string data)
    {
        byte[] myData = Encoding.UTF8.GetBytes(data);

        using (UnityWebRequest request = UnityWebRequest.Put(url, data))
        {
            yield return request.SendWebRequest();

            if (request.isNetworkError || request.isHttpError)
            {
                UnityEngine.Debug.Log(request.error);
            }

            if (request.isDone == true)
            {
                UnityEngine.Debug.Log("Upload complete!");
                loggerData = "";            // reset the logger data being sent to Python server
            }
        } 
    }
}
