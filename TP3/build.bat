@echo off
set PARAMS=%~1
set TARGETS=exo1,exo2,exo3,exo4,exo5
set VCINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 16.0\VC
setlocal

call :initColorPrint

if not exist win32 (
    call :colorPrint 4e , "Creates the win32 directory" , /n
    mkdir win32
)
call :colorPrint 4e , "Move into ./win32" , /n
cd win32
call :colorPrint 4e , "Configure the projet ..." , /n
REM cmake -G"Visual Studio 17 2022" ..
cmake -G"Visual Studio 16 2019" ..

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
cmake --build . --config Release --target %~1 -- /verbosity:quiet
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
