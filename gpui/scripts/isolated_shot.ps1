# Run an exe on a separate, invisible Windows desktop and capture its main window with PrintWindow.
# Nothing touches the user's desktop, cursor, keyboard, focus or clipboard.
param([string]$Exe, [string]$WorkDir, [string]$Out, [int]$WaitMs = 6000, [string]$ExtraArgs = '')
Add-Type -ReferencedAssemblies System.Drawing @'
using System; using System.Drawing; using System.Runtime.InteropServices; using System.Text;
public static class Iso {
  [DllImport("user32.dll", SetLastError=true)] public static extern IntPtr CreateDesktop(string name, IntPtr dev, IntPtr mode, int flags, uint access, IntPtr sa);
  [DllImport("user32.dll")] public static extern bool CloseDesktop(IntPtr h);
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumDesktopWindows(IntPtr desk, EnumProc f, IntPtr l);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] public struct STARTUPINFO {
    public int cb; public string lpReserved; public string lpDesktop; public string lpTitle; public int dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags; public short wShowWindow, cbReserved2; public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError; }
  public struct PROCESS_INFORMATION { public IntPtr hProcess, hThread; public int dwProcessId, dwThreadId; }
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] public static extern bool CreateProcess(string app, string cmd, IntPtr pa, IntPtr ta, bool inherit, uint flags, IntPtr env, string dir, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
  [DllImport("kernel32.dll")] public static extern bool TerminateProcess(IntPtr h, uint code);
  public static IntPtr FindMain(IntPtr desk, int pid) {
    IntPtr best = IntPtr.Zero; int area = 0;
    EnumDesktopWindows(desk, (h, l) => { uint p; GetWindowThreadProcessId(h, out p); RECT r; GetWindowRect(h, out r);
      int a = (r.R - r.L) * (r.B - r.T); if (p == pid && IsWindowVisible(h) && a > area) { area = a; best = h; } return true; }, IntPtr.Zero);
    return best;
  }
  public static bool Shot(IntPtr h, string path) {
    RECT r; GetWindowRect(h, out r); int w = r.R - r.L, hh = r.B - r.T; if (w <= 0 || hh <= 0) return false;
    using (var bmp = new Bitmap(w, hh)) { using (var g = Graphics.FromImage(bmp)) { IntPtr dc = g.GetHdc(); PrintWindow(h, dc, 2); g.ReleaseHdc(dc); } bmp.Save(path); }
    return true;
  }
}
'@
[Iso]::SetProcessDPIAware() | Out-Null
$name = 'dotsmith-test-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$desk = [Iso]::CreateDesktop($name, [IntPtr]::Zero, [IntPtr]::Zero, 0, 0x01FF, [IntPtr]::Zero)
if ($desk -eq [IntPtr]::Zero) { throw "CreateDesktop failed" }
$si = New-Object Iso+STARTUPINFO; $si.cb = [Runtime.InteropServices.Marshal]::SizeOf($si); $si.lpDesktop = $name
$pi = New-Object Iso+PROCESS_INFORMATION
$cmd = '"' + $Exe + '" ' + $ExtraArgs
$env:DOTSMITH_MULTI_INSTANCE = '1'
if (-not [Iso]::CreateProcess($Exe, $cmd, [IntPtr]::Zero, [IntPtr]::Zero, $false, 0, [IntPtr]::Zero, $WorkDir, [ref]$si, [ref]$pi)) {
  $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error(); [Iso]::CloseDesktop($desk) | Out-Null; throw "CreateProcess failed: $err" }
try {
  Start-Sleep -Milliseconds $WaitMs
  $h = [Iso]::FindMain($desk, $pi.dwProcessId)
  if ($h -eq [IntPtr]::Zero) { "no window found" } elseif ([Iso]::Shot($h, $Out)) { "captured $Out" } else { "capture failed" }
} finally {
  [Iso]::TerminateProcess($pi.hProcess, 0) | Out-Null
  Start-Sleep -Milliseconds 300
  [Iso]::CloseDesktop($desk) | Out-Null
}
