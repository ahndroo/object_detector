# To mount camera from WSL: https://learn.microsoft.com/en-us/windows/wsl/connect-usb#attach-a-usb-device
# In privileged powershell run:
winget install dorssel.usbipd-win
usbipd bind --busid <busid>
usbipd attach --wsl --busid <busid>

##############
https://obsproject.com/
https://github.com/iamscottxu/obs-rtspserver/releases
