; Inno Setup script — optional single-installer packaging of the portable bundle.
; Build the portable folder first (packaging\build_windows.ps1), then:
;   1. install Inno Setup 6 or 7 (https://jrsoftware.org/isinfo.php)
;   2. compile this file with ISCC.exe or the Inno Studio GUI (Build → Compile / F9)
;
; Output: dist\installer\GBVMonitor-setup.exe — for bundles > ~1.9 GB the output is
; disk-spanned, i.e. GBVMonitor-setup.exe plus GBVMonitor-setup-2.bin, -3.bin, …
; simply keep ALL parts in ONE folder; the user runs the .exe and the rest is read
; from the .bin files. NOTICE: the user installing via the setup gets the same
; folder-extraction experience as unzipping the portable zip — no admin needed,
; no admin-required system directories touched, the app needs no registry entries.

[Setup]
; Sources resolve against the PROJECT ROOT (script lives in packaging\), no matter
; how ISCC is invoked (GUI or CLI).
SourceDir=..
; Installer + uninstaller icon (CWD-independent: {#SourcePath} = this .iss folder).
; Shortcut icons inherit the app icon from GBVMonitor.exe (set in gbv_monitor.spec).
SetupIconFile={#SourcePath}\icon.ico
AppName=GBV Monitor
AppVersion=1.0.0
AppCopyright=For civil-society / NGO use
; {autopf} = "Program Files" for admins, the per-user Programs folder otherwise —
; always writable under PrivilegesRequired=lowest (C:\ root would not be).
DefaultDirName={autopf}\GBVMonitor
DefaultGroupName=GBV Monitor
Compression=lzma2/max
DiskSpanning=yes
DiskSliceSize=2000000000
OutputDir=dist\installer
OutputBaseFilename=GBVMonitor-setup
Uninstallable=yes
UninstallFilesDir={app}\uninst
PrivilegesRequired=lowest

[Files]
Source: "dist\GBVMonitor\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\GBV Monitor"; Filename: "{app}\GBVMonitor.exe"
Name: "{userdesktop}\GBV Monitor"; Filename: "{app}\GBVMonitor.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Dirs]
Name: "{app}\output"; Permissions: users-modify
Name: "{app}\logs";   Permissions: users-modify

[UninstallDelete]
; keep the user's case dataset safe: only uninstall files, not output\
Type: files; Name: "{app}\uninst\*"
