@echo off
setlocal

cmake -S . -B ./build -DCMAKE_TOOLCHAIN_FILE=C:/Users/Acer/Documents/imgproc_s2025/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build ./build
set /p exePath="Task name: "

if not exitst "./bin.dbg/%exePath%" (
    echo Файл не найдет: %exePath%
    goto :eof
)

set /p args="args (default=''): "

echo Run %exePath%" %args%

start "" "./bin.dbg/%exePath%" %args%

:end