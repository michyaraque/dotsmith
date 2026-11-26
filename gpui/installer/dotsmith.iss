; Dotsmith installer (Inno Setup 6). Built by scripts\package.ps1, which passes AppVersion and SourceDir.
#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\Dotsmith"
#endif

[Setup]
AppId={{6F0C2B9E-5A41-4C7E-9F0B-3E1D2A7C8B55}
AppName=Dotsmith
AppVersion={#AppVersion}
AppPublisher=Michael Araque
AppPublisherURL=https://github.com/michyaraque
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\Dotsmith
DefaultGroupName=Dotsmith
DisableProgramGroupPage=yes
OutputBaseFilename=Dotsmith-{#AppVersion}-setup
SetupIconFile=..\assets\dotsmith.ico
LicenseFile=..\..\LICENSE
UninstallDisplayIcon={app}\dotsmith.exe
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\Dotsmith"; Filename: "{app}\dotsmith.exe"
Name: "{autodesktop}\Dotsmith"; Filename: "{app}\dotsmith.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\dotsmith.exe"; Description: "{cm:LaunchProgram,Dotsmith}"; Flags: nowait postinstall skipifsilent
