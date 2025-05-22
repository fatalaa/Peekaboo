#!/usr/bin/osascript
--------------------------------------------------------------------------------
-- peekaboo.scpt - v1.0.0 "Peekaboo! 👀 → 📸 → 💾"
-- Unattended screenshot capture with app targeting and location specification
-- Peekaboo—screenshot got you! Now you see it, now it's saved.
--------------------------------------------------------------------------------

--#region Configuration Properties
property scriptInfoPrefix : "Peekaboo 👀: "
property defaultScreenshotFormat : "png"
property captureDelay : 1.0 -- Delay after bringing app to front before capture
property windowActivationDelay : 0.5 -- Delay for window activation
property enhancedErrorReporting : true
property verboseLogging : false
--#endregion Configuration Properties

--#region Helper Functions
on isValidPath(thePath)
    if thePath is not "" and (thePath starts with "/") then
        return true
    end if
    return false
end isValidPath

on getFileExtension(filePath)
    set oldDelims to AppleScript's text item delimiters
    set AppleScript's text item delimiters to "."
    set pathParts to text items of filePath
    set AppleScript's text item delimiters to oldDelims
    if (count pathParts) > 1 then
        return item -1 of pathParts
    else
        return ""
    end if
end getFileExtension

on ensureDirectoryExists(dirPath)
    try
        do shell script "mkdir -p " & quoted form of dirPath
        return true
    on error
        return false
    end try
end ensureDirectoryExists

on formatErrorMessage(errorType, errorMsg, context)
    if enhancedErrorReporting then
        set formattedMsg to scriptInfoPrefix & errorType & ": " & errorMsg
        if context is not "" then
            set formattedMsg to formattedMsg & " (Context: " & context & ")"
        end if
        return formattedMsg
    else
        return scriptInfoPrefix & errorMsg
    end if
end formatErrorMessage

on logVerbose(message)
    if verboseLogging then
        log "🔍 " & message
    end if
end logVerbose

on trimWhitespace(theText)
    set whitespaceChars to {" ", tab}
    set newText to theText
    repeat while (newText is not "") and (character 1 of newText is in whitespaceChars)
        if (length of newText) > 1 then
            set newText to text 2 thru -1 of newText
        else
            set newText to ""
        end if
    end repeat
    repeat while (newText is not "") and (character -1 of newText is in whitespaceChars)
        if (length of newText) > 1 then
            set newText to text 1 thru -2 of newText
        else
            set newText to ""
        end if
    end repeat
    return newText
end trimWhitespace
--#endregion Helper Functions

--#region App Resolution Functions
on resolveAppIdentifier(appIdentifier)
    my logVerbose("Resolving app identifier: " & appIdentifier)
    
    -- First try as bundle ID
    try
        tell application "System Events"
            set bundleApps to (every application process whose bundle identifier is appIdentifier)
            if (count bundleApps) > 0 then
                set targetApp to item 1 of bundleApps
                set appName to name of targetApp
                my logVerbose("Found running app by bundle ID: " & appName)
                return {appName:appName, bundleID:appIdentifier, isRunning:true, resolvedBy:"bundle_id"}
            end if
        end tell
    on error
        my logVerbose("Bundle ID lookup failed, trying as app name")
    end try
    
    -- Try as application name for running apps
    try
        tell application "System Events"
            set nameApps to (every application process whose name is appIdentifier)
            if (count nameApps) > 0 then
                set targetApp to item 1 of nameApps
                try
                    set bundleID to bundle identifier of targetApp
                on error
                    set bundleID to ""
                end try
                my logVerbose("Found running app by name: " & appIdentifier)
                return {appName:appIdentifier, bundleID:bundleID, isRunning:true, resolvedBy:"app_name"}
            end if
        end tell
    on error
        my logVerbose("App name lookup failed for running processes")
    end try
    
    -- Try to find the app in /Applications (not running)
    try
        set appPath to "/Applications/" & appIdentifier & ".app"
        tell application "System Events"
            if exists file appPath then
                try
                    set bundleID to bundle identifier of file appPath
                on error
                    set bundleID to ""
                end try
                my logVerbose("Found app in /Applications: " & appIdentifier)
                return {appName:appIdentifier, bundleID:bundleID, isRunning:false, resolvedBy:"applications_folder"}
            end if
        end tell
    on error
        my logVerbose("/Applications lookup failed")
    end try
    
    -- If it looks like a bundle ID, try launching it directly
    if appIdentifier contains "." then
        try
            tell application "System Events"
                launch application id appIdentifier
                delay windowActivationDelay
                set bundleApps to (every application process whose bundle identifier is appIdentifier)
                if (count bundleApps) > 0 then
                    set targetApp to item 1 of bundleApps
                    set appName to name of targetApp
                    my logVerbose("Successfully launched app by bundle ID: " & appName)
                    return {appName:appName, bundleID:appIdentifier, isRunning:true, resolvedBy:"bundle_id_launch"}
                end if
            end tell
        on error errMsg
            my logVerbose("Bundle ID launch failed: " & errMsg)
        end try
    end if
    
    return missing value
end resolveAppIdentifier

on bringAppToFront(appInfo)
    set appName to appName of appInfo
    set isRunning to isRunning of appInfo
    
    my logVerbose("Bringing app to front: " & appName & " (running: " & isRunning & ")")
    
    if not isRunning then
        try
            tell application appName to activate
            delay windowActivationDelay
        on error errMsg
            return my formatErrorMessage("Activation Error", "Failed to launch app '" & appName & "': " & errMsg, "app launch")
        end try
    else
        try
            tell application "System Events"
                tell process appName
                    set frontmost to true
                end tell
            end tell
            delay windowActivationDelay
        on error errMsg
            return my formatErrorMessage("Focus Error", "Failed to bring app '" & appName & "' to front: " & errMsg, "app focus")
        end try
    end if
    
    return ""
end bringAppToFront
--#endregion App Resolution Functions

--#region Screenshot Functions
on captureScreenshot(outputPath)
    my logVerbose("Capturing screenshot to: " & outputPath)
    
    -- Ensure output directory exists
    set outputDir to do shell script "dirname " & quoted form of outputPath
    if not my ensureDirectoryExists(outputDir) then
        return my formatErrorMessage("Directory Error", "Could not create output directory: " & outputDir, "directory creation")
    end if
    
    -- Wait for capture delay
    delay captureDelay
    
    -- Determine screenshot format
    set fileExt to my getFileExtension(outputPath)
    if fileExt is "" then
        set fileExt to defaultScreenshotFormat
        set outputPath to outputPath & "." & fileExt
    end if
    
    -- Capture screenshot using screencapture
    try
        set screencaptureCmd to "screencapture -x"
        
        -- Add format flag if not PNG (default)
        if fileExt is not "png" then
            set screencaptureCmd to screencaptureCmd & " -t " & fileExt
        end if
        
        -- Add output path
        set screencaptureCmd to screencaptureCmd & " " & quoted form of outputPath
        
        my logVerbose("Running: " & screencaptureCmd)
        do shell script screencaptureCmd
        
        -- Verify file was created
        try
            do shell script "test -f " & quoted form of outputPath
            return outputPath
        on error
            return my formatErrorMessage("Capture Error", "Screenshot file was not created at: " & outputPath, "file verification")
        end try
        
    on error errMsg number errNum
        return my formatErrorMessage("Capture Error", "screencapture failed: " & errMsg, "error " & errNum)
    end try
end captureScreenshot
--#endregion Screenshot Functions

--#region Main Script Logic (on run)
on run argv
    set appSpecificErrorOccurred to false
    try
        my logVerbose("Starting Screenshotter v1.0.0")
        
        set argCount to count argv
        if argCount < 2 then return my usageText()
        
        set appIdentifier to item 1 of argv
        set outputPath to item 2 of argv
        
        -- Validate arguments
        if appIdentifier is "" then
            return my formatErrorMessage("Argument Error", "App identifier cannot be empty." & linefeed & linefeed & my usageText(), "validation")
        end if
        
        if not my isValidPath(outputPath) then
            return my formatErrorMessage("Argument Error", "Output path must be an absolute path starting with '/'." & linefeed & linefeed & my usageText(), "validation")
        end if
        
        -- Resolve app identifier
        set appInfo to my resolveAppIdentifier(appIdentifier)
        if appInfo is missing value then
            return my formatErrorMessage("Resolution Error", "Could not resolve app identifier '" & appIdentifier & "'. Check that the app name or bundle ID is correct.", "app resolution")
        end if
        
        set resolvedAppName to appName of appInfo
        set resolvedBy to resolvedBy of appInfo
        my logVerbose("App resolved: " & resolvedAppName & " (method: " & resolvedBy & ")")
        
        -- Bring app to front
        set frontError to my bringAppToFront(appInfo)
        if frontError is not "" then return frontError
        
        -- Capture screenshot
        set screenshotResult to my captureScreenshot(outputPath)
        if screenshotResult starts with scriptInfoPrefix then
            -- Error occurred
            return screenshotResult
        else
            -- Success
            return scriptInfoPrefix & "Screenshot captured successfully: " & screenshotResult & " (App: " & resolvedAppName & ")"
        end if
        
    on error generalErrorMsg number generalErrorNum
        if appSpecificErrorOccurred then error generalErrorMsg number generalErrorNum
        return my formatErrorMessage("Execution Error", generalErrorMsg, "error " & generalErrorNum)
    end try
end run
--#endregion Main Script Logic (on run)

--#region Usage Function
on usageText()
    set LF to linefeed
    set scriptName to "peekaboo.scpt"
    
    set outText to scriptName & " - v1.0.0 \"Peekaboo! 👀 → 📸 → 💾\" – AppleScript Screenshot Utility" & LF & LF
    set outText to outText & "Peekaboo—screenshot got you! Now you see it, now it's saved." & LF
    set outText to outText & "Takes unattended screenshots of applications by name or bundle ID." & LF & LF
    
    set outText to outText & "Usage:" & LF
    set outText to outText & "  osascript " & scriptName & " \"<app_name_or_bundle_id>\" \"<output_path>\"" & LF & LF
    
    set outText to outText & "Parameters:" & LF
    set outText to outText & "  app_name_or_bundle_id: Application name (e.g., 'Safari') or bundle ID (e.g., 'com.apple.Safari')" & LF
    set outText to outText & "  output_path:          Absolute path for screenshot file (e.g., '/Users/name/Desktop/screenshot.png')" & LF & LF
    
    set outText to outText & "Examples:" & LF
    set outText to outText & "  # Screenshot Safari using app name:" & LF
    set outText to outText & "  osascript " & scriptName & " \"Safari\" \"/Users/username/Desktop/safari_shot.png\"" & LF
    set outText to outText & "  # Screenshot using bundle ID:" & LF
    set outText to outText & "  osascript " & scriptName & " \"com.apple.TextEdit\" \"/tmp/textedit.png\"" & LF
    set outText to outText & "  # Screenshot with different format:" & LF
    set outText to outText & "  osascript " & scriptName & " \"Xcode\" \"/Users/username/Screenshots/xcode.jpg\"" & LF & LF
    
    set outText to outText & "Features:" & LF
    set outText to outText & "  • Automatically resolves app names to bundle IDs and vice versa" & LF
    set outText to outText & "  • Launches apps if not running" & LF
    set outText to outText & "  • Brings target app to front before capture" & LF
    set outText to outText & "  • Supports PNG, JPG, PDF, and other formats via file extension" & LF
    set outText to outText & "  • Creates output directories automatically" & LF
    set outText to outText & "  • Enhanced error reporting with context" & LF & LF
    
    set outText to outText & "Notes:" & LF
    set outText to outText & "  • Requires Screen Recording permission in System Preferences > Security & Privacy" & LF
    set outText to outText & "  • Output path must be absolute (starting with '/')" & LF
    set outText to outText & "  • Default format is PNG if no extension specified" & LF
    set outText to outText & "  • The script will wait " & (captureDelay as string) & " second(s) after bringing app to front before capture" & LF
    
    return outText
end usageText
--#endregion Usage Function