@echo off
set PARAMS=%~1
set TARGETS=PremierPas,Communication,NaiveBroadcast,PipeBroadcast
set VCINSTALLDIR=C:\Program Files\Microsoft Visual Studio\2022\Community\VC
setlocal

call :initColorPrint

if not exist win32 (
    call :colorPrint 4e , "Creates the win32 directory" , /n
    mkdir win32
)
call :colorPrint 4e , "Move into ./win32" , /n
cd win32
call :colorPrint 4e , "Configure the projet ..." , /n
cmake -G"Visual Studio 17 2022" ..

if not [%PARAMS%]==[] set TARGETS=%PARAMS%

call :colorPrint 4e , "Build the project ..." , /n
for %%I IN (%TARGETS%) DO (
    call :buildTarget %%I
)

REM bye bye
call :colorPrint 4e , "That's all, folks!" , /n
call :cleanupColorPrint
exit /b


REM This function build one target
:buildTarget target
call :colorPrint 4e , "Build TARGET %~1:" , /n
cmake --build . --config Release --target %~1 -- /verbosity:normal
exit /b

REM This function print a string, and may be a newline
:colorPrint _col _str [/n]
setlocal
set "str=%~2"
call :colorPrintVar %~1 str %~3
exit /b

REM this function is used to print a string
:colorPrintVar _col _strVar [/n]
if not defined %~2 exit /b
setlocal enableDelayedExpansion
set "str=a%DEL%!%~2:\=a%DEL%\..\%DEL%%DEL%%DEL%!"
set "str=!str:/=a%DEL%/..\%DEL%%DEL%%DEL%!"
set "str=!str:"=\"!"
pushd "%temp%"
findstr /p /A:%~1 "." "!str!\..\x" nul
if /i "%~3"=="/n" echo(
exit /b

REM this function initializes the color print system
:initColorPrint
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "DEL=%%a"
<nul >"%temp%\x" set /p "=%DEL%%DEL%%DEL%%DEL%%DEL%%DEL%.%DEL%"
exit /b

REM this function ends the color print system
:cleanupColorPrint
del "%temp%\x"
exit /b
