// Rula.exe / RulaCoder.exe — starts the selected Rula desktop app.
//
// Built by build.ps1 with the C# compiler that ships with Windows (.NET
// Framework 4), so building needs nothing installed and takes about a second.
// Written in C# 5 for that compiler: no string interpolation, no newer syntax.
//
// What it does:
//   1. starts the backend (pythonw -m src.rula.gui) on a free 127.0.0.1 port,
//      with a random token only this window knows
//   2. opens it as an app window in Edge (no address bar, own taskbar entry)
//   3. when that window closes, stops the backend — which takes llama-server and
//      its VRAM down with it
// Reopening the same app shows its window; launching the other app asks before
// signaling the current process to close and handing off the shared GPU lock.
//
// It can live anywhere — inside the Rula folder, or on the desktop. build.ps1
// writes the folder it was built in into InstallDir.g.cs (InstallInfo.Dir), and
// Rula's own Python under runtime\python is used ahead of any installed one.
using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading;
using System.Windows.Forms;

#if CODER_APP
[assembly: AssemblyTitle("RulaCoder")]
[assembly: AssemblyProduct("RulaCoder")]
[assembly: AssemblyDescription("RulaCoder — 自律型コーディングAI")]
#else
[assembly: AssemblyTitle("Rula")]
[assembly: AssemblyProduct("Rula")]
[assembly: AssemblyDescription("Rula — 日本語AIアシスタント")]
#endif
[assembly: AssemblyVersion("2.0.0.0")]
[assembly: AssemblyFileVersion("2.0.0.0")]

static class RulaLauncher
{
    const string MutexName = "Local\\RulaDesktopApp";
    const string SwitchEventName = "Local\\RulaDesktopAppSwitch";
#if CODER_APP
    const string AppMode = "coder";
    const string AppName = "RulaCoder";
    const string OtherAppName = "Rula";
#else
    const string AppMode = "general";
    const string AppName = "Rula";
    const string OtherAppName = "RulaCoder";
#endif

    [STAThread]
    static int Main()
    {
        string app = FindApp();
        if (app == null)
        {
            Fail(AppName + " の本体が見つかりません。\nセットアップをもう一度実行してください。\n(探した場所: " + InstallInfo.Dir + ")");
            return 1;
        }
        string data = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Rula");
        Directory.CreateDirectory(data);
        string runFile = Path.Combine(data, "running.txt");

        bool first;
        using (Mutex mutex = new Mutex(true, MutexName, out first))
        using (EventWaitHandle switchEvent = new EventWaitHandle(false, EventResetMode.AutoReset, SwitchEventName))
        {
            if (!first)
            {
                string activeMode, activeUrl;
                if (!ReadRunningApp(runFile, out activeMode, out activeUrl))
                {
                    bool recovered = false;
                    try { recovered = mutex.WaitOne(0); }
                    catch (AbandonedMutexException) { recovered = true; }
                    if (!recovered)
                    {
                        Fail("起動中のアプリを確認できませんでした。\n少し待ってから、もう一度起動してください。");
                        return 1;
                    }
                }
                else if (activeMode == AppMode)
                {
                    if (BackendAlive(activeUrl)) OpenWindow(activeUrl);
                    return 0;
                }
                else if (activeMode == "general" || activeMode == "coder")
                {
                    DialogResult choice = MessageBox.Show(
                        OtherAppName + " が起動中です。\n\n" +
                        "「はい」: " + OtherAppName + " を終了して " + AppName + " を起動\n" +
                        "「いいえ」: " + AppName + " の起動をやめて " + OtherAppName + " を使い続ける",
                        AppName + " の起動確認", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                    if (choice != DialogResult.Yes) return 0;

                    switchEvent.Set();
                    bool acquired = false;
                    try { acquired = mutex.WaitOne(TimeSpan.FromSeconds(30)); }
                    catch (AbandonedMutexException) { acquired = true; }
                    if (!acquired)
                    {
                        Fail(OtherAppName + " を終了できませんでした。\n起動中のアプリを閉じてから、もう一度お試しください。");
                        return 1;
                    }
                }
            }

            switchEvent.WaitOne(0);
            int port = FreePort();
            string token = Token();
            string url = "http://127.0.0.1:" + port + "/?t=" + token;

            Process backend = StartBackend(app, port, token);
            if (backend == null) return 1;
            Process window = null;
            bool switching = false;
            DateTime switchDeadline = DateTime.MaxValue;
            try
            {
                File.WriteAllText(runFile, AppMode + "|" + url);
                bool switchDuringStartup;
                if (!WaitForBackend(port, backend, 90, switchEvent, out switchDuringStartup))
                {
                    switching = switchDuringStartup;
                    if (!switching)
                        Fail(AppName + " を起動できませんでした。\nログ: " +
                             Path.Combine(Environment.GetFolderPath(
                                 Environment.SpecialFolder.ApplicationData),
                                 AppMode == "coder" ? "RulaCoder" : "Rula", "rula.log"));
                    return 1;
                }
                window = OpenWindow(url);
                while (!backend.HasExited && (window == null || !window.HasExited))
                {
                    if (switchEvent.WaitOne(0))
                    {
                        switching = true;
                        switchDeadline = DateTime.Now.AddSeconds(15);
                        RequestSwitch(url);
                    }
                    if (switching && DateTime.Now >= switchDeadline) break;
                    Thread.Sleep(250);
                }
            }
            finally
            {
                if (switching && window != null)
                    KillProcessTree(window);
                try
                {
                    if (!backend.HasExited) backend.Kill();
                }
                catch (Exception) { }
                try
                {
                    if (File.Exists(runFile) && File.ReadAllText(runFile).StartsWith(AppMode + "|"))
                        File.Delete(runFile);
                }
                catch (Exception) { }
            }
        }
        return 0;
    }

    static bool ReadRunningApp(string runFile, out string mode, out string url)
    {
        mode = "";
        url = "";
        try
        {
            string value = File.ReadAllText(runFile).Trim();
            int separator = value.IndexOf('|');
            if (separator < 0)
            {
                // Compatibility with a Rula process started by an older build.
                mode = "general";
                url = value;
            }
            else
            {
                mode = value.Substring(0, separator);
                url = value.Substring(separator + 1);
            }
            Uri parsed;
            return (mode == "general" || mode == "coder") &&
                   Uri.TryCreate(url, UriKind.Absolute, out parsed) &&
                   parsed.IsLoopback && parsed.Scheme == Uri.UriSchemeHttp;
        }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
    }

    // The Rula folder: next to this exe, else where it was built, else the
    // one-line installer's default location.
    static string FindApp()
    {
        string[] candidates = {
            AppDomain.CurrentDomain.BaseDirectory,
            InstallInfo.Dir,
            Environment.GetEnvironmentVariable("RULA_HOME"),
            Path.Combine(Environment.GetFolderPath(
                Environment.SpecialFolder.LocalApplicationData), "Programs", "Rula") };
        foreach (string dir in candidates)
        {
            if (!string.IsNullOrEmpty(dir) && Directory.Exists(Path.Combine(dir, "src", "rula")))
                return dir;
        }
        return null;
    }

    static Process StartBackend(string app, int port, string token)
    {
        string python = FindPython(app);
        if (python == null)
        {
            Fail("Python が見つかりません。Python 3 をインストールするか、環境変数 RULA_PYTHON に python.exe の場所を設定してください。");
            return null;
        }
        ProcessStartInfo psi = new ProcessStartInfo(python,
            "-m src.rula.gui --port " + port + " --token " + token);
        psi.WorkingDirectory = app;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["RULA_APP_MODE"] = AppMode;
        return Process.Start(psi);
    }

    static void RequestSwitch(string url)
    {
        try
        {
            Uri page = new Uri(url);
            string endpoint = "http://" + page.Authority + "/api/switch" + page.Query;
            HttpWebRequest req = (HttpWebRequest)WebRequest.Create(endpoint);
            req.Method = "POST";
            req.ContentLength = 0;
            req.Timeout = 1500;
            req.Proxy = null;
            using (HttpWebResponse response = (HttpWebResponse)req.GetResponse()) { }
        }
        catch (WebException) { }
        catch (UriFormatException) { }
    }

    static void KillProcessTree(Process process)
    {
        try
        {
            if (process.HasExited) return;
            ProcessStartInfo psi = new ProcessStartInfo(
                "taskkill.exe", "/PID " + process.Id + " /T /F");
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            using (Process killer = Process.Start(psi)) killer.WaitForExit(5000);
        }
        catch (Exception) { }
    }

    static string FindPython(string app)
    {
        // Rula's own Python (install.ps1): the packages Rula was tested with.
        string own = Path.Combine(app, "runtime", "python", "pythonw.exe");
        if (File.Exists(own)) return own;
        string env = Environment.GetEnvironmentVariable("RULA_PYTHON");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string path = Environment.GetEnvironmentVariable("PATH") ?? "";
        // pythonw first: python.exe would still own a (hidden) console.
        foreach (string name in new string[] { "pythonw.exe", "python.exe" })
        {
            foreach (string dir in path.Split(';'))
            {
                if (dir.Length == 0) continue;
                // The Store's python.exe alias opens the Store instead of running.
                if (dir.IndexOf("WindowsApps", StringComparison.OrdinalIgnoreCase) >= 0) continue;
                try
                {
                    string candidate = Path.Combine(dir.Trim(), name);
                    if (File.Exists(candidate)) return candidate;
                }
                catch (ArgumentException) { }
            }
        }
        return null;
    }

    static bool BackendAlive(string url)
    {
        try
        {
            Uri page = new Uri(url);
            HttpWebRequest req = (HttpWebRequest)WebRequest.Create(
                "http://" + page.Authority + "/");
            req.Timeout = 500;
            req.Proxy = null;
            using (HttpWebResponse response = (HttpWebResponse)req.GetResponse())
                return response.StatusCode == HttpStatusCode.OK;
        }
        catch (WebException) { return false; }
        catch (UriFormatException) { return false; }
    }

    static bool WaitForBackend(int port, Process backend, int seconds,
                               EventWaitHandle switchEvent, out bool switching)
    {
        switching = false;
        DateTime deadline = DateTime.Now.AddSeconds(seconds);
        while (DateTime.Now < deadline)
        {
            if (switchEvent.WaitOne(0))
            {
                switching = true;
                return false;
            }
            if (backend.HasExited) return false;
            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + port + "/");
                req.Timeout = 1000;
                req.Proxy = null;
                using (HttpWebResponse res = (HttpWebResponse)req.GetResponse())
                {
                    if (res.StatusCode == HttpStatusCode.OK) return true;
                }
            }
            catch (WebException) { }
            Thread.Sleep(250);
        }
        return false;
    }

    static Process OpenWindow(string url)
    {
        string edge = FindEdge();
        if (edge == null)
        {
            Process.Start(url);  // default browser; no process to wait on
            return null;
        }
        // A profile of its own, so it shares nothing with the user's browsing.
        // Extensions and sync are off: Edge signs a new profile into the Windows
        // account and syncs extensions into it, and one of them (MetaMask) opened
        // a window of its own there. Same flags as gui/server.py EDGE_FLAGS.
        string profile = Path.Combine(Environment.GetFolderPath(
            Environment.SpecialFolder.LocalApplicationData), "Rula",
            AppMode == "coder" ? "RulaCoderEdgeProfile" : "EdgeProfile");
        string args = "--app=\"" + url + "\" --user-data-dir=\"" + profile + "\"" +
                      " --window-size=1180,820 --no-first-run --no-default-browser-check" +
                      " --disable-extensions --disable-sync" +
                      " --disable-features=Translate,msEdgeSidebarV2";
        Process p = Process.Start(new ProcessStartInfo(edge, args) { UseShellExecute = false });
        // If Edge passed the window to an instance that was already running with
        // this profile, this process ends at once and is not the window.
        if (p != null && p.WaitForExit(1500)) return null;
        return p;
    }

    static string FindEdge()
    {
        string[] roots = {
            Environment.GetEnvironmentVariable("ProgramFiles(x86)"),
            Environment.GetEnvironmentVariable("ProgramFiles"),
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) };
        foreach (string root in roots)
        {
            if (string.IsNullOrEmpty(root)) continue;
            string p = Path.Combine(root, "Microsoft", "Edge", "Application", "msedge.exe");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    static int FreePort()
    {
        TcpListener l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        int port = ((IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        return port;
    }

    static string Token()
    {
        byte[] b = new byte[24];
        using (RNGCryptoServiceProvider rng = new RNGCryptoServiceProvider()) rng.GetBytes(b);
        return BitConverter.ToString(b).Replace("-", "").ToLowerInvariant();
    }

    static void Fail(string message)
    {
        MessageBox.Show(message, "Rula", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }
}
