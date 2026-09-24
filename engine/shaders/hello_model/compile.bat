@echo off
set GLSLC=C:\VulkanSDK\1.3.268.0\Bin\glslc.exe

"%GLSLC%" "%~dp0\shader.vert" -o "%~dp0\vert.spv"
"%GLSLC%" "%~dp0\shader.frag" -o "%~dp0\frag.spv"
"%GLSLC%" "%~dp0\shader.comp" -o "%~dp0\comp.spv"
pause