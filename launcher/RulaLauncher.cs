// Rula.exe — starts Rula and opens its window.
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
// Launched again while running, it just opens another window onto the same app.
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

[assembly: AssemblyTitle("Rula")]
[assembly: AssemblyProduct("Rula")]
[assembly: AssemblyDescription("Rula — 日本語AIアシスタント")]
[assembly: AssemblyVersion("2.0.0.0")]
[assembly: AssemblyFileVersion("2.0.0.0")]

static class RulaLauncher
{
    [STAThread]
    static int Main()
    {
        string app = FindApp();
        if (app == null)
        {
            Fail("Rula の本体が見つかりません。\nセットアップをもう一度実行してください。\n(探した場所: " + InstallInfo.Dir + ")");
            return 1;
        }
        string data = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Rula");
        Directory.CreateDirectory(data);
        string runFile = Path.Combine(data, "running.txt");

        bool first;
        using (Mutex mutex = new Mutex(true, "Local\\RulaDesktopApp", out first))
        {
            if (!first)
            {
                // Already running: another window onto the same backend.
                if (File.Exists(runFile)) OpenWindow(File.ReadAllText(runFile).Trim());
                return 0;
            }

            int port = FreePort();
            string token = Token();
            string url = "http://127.0.0.1:" + port + "/?t=" + token;

            Process backend = StartBackend(app, port, token);
            if (backend == null) return 1;
            try
            {
                if (!WaitForBackend(port, backend, 90))
                {
                    Fail("Rula を起動できませんでした。\nログ: " + Path.Combine(data, "rula.log"));
                    return 1;
                }
                File.WriteAllText(runFile, url);
                Process window = OpenWindow(url);
                if (window != null)
                {
                    // Either one ending means the app is done: the backend quits
                    // itself when the page signals that its window closed, even if
                    // this Edge process stays up holding other windows.
                    while (!window.HasExited && !backend.HasExited) Thread.Sleep(500);
                }
                // No window process to wait on (default browser, or Edge handed the
                // window to an instance already running): the backend exits by
                // itself once the page stops sending its heartbeat.
                else
                {
                    backend.WaitForExit();
                }
            }
            finally
            {
                try { File.Delete(runFile); } catch (Exception) { }
                try { if (!backend.HasExited) backend.Kill(); } catch (Exception) { }
            }
        }
        return 0;
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
        return Process.Start(psi);
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

    static bool WaitForBackend(int port, Process backend, int seconds)
    {
        DateTime deadline = DateTime.Now.AddSeconds(seconds);
        while (DateTime.Now < deadline)
        {
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
            Environment.SpecialFolder.LocalApplicationData), "Rula", "EdgeProfile");
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
