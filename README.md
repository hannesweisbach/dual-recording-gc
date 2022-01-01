# dual-recording-gc
Automate transferring fit-Files from Garmin Connect to ZwiftPower for dual-recording analysis

## Requirements

* Python3
* Python packages: cloudscraper, fitparse, garminconnect, lxml, numpy, pandas, requests

## Usage

The script is simply run by executing it: `./dual-recording-gc.py` or `python3 dual-recording-gc.py`.

## Function & Features

`dual-recording-gc.py` first connects to ZwiftPower, collecting a list of all Zwift activities ZwiftPower knows about and selects the latest one. Next, the script connects to Garmin Connect and attempts to find and download an activity with the same date as the activity on ZwiftPower. Next, `dual-recording-gc.py` reads the two fit-files and attempts to align the power curves[^1] and to identify the power source used. Next, the fit-file downloaded from Garmin Connect is uploaded to ZwiftPower. Both ZwiftPower and Garmin Connect fit files power source names are added. Subsequently, the time offset of the Garmin Connect fit-file is adjusted to the previously calculated alignment value.

[^1]: Aligning the offsets can be necessary, for example, when Garmin devices don't have a GPS-lock for an extended period of time. Garmin devices update their time only by GPS-time, never over NTP, even when WiFi or Bluetooth is enabled and connected.

## Configuration File `config.json`

A configuration file `config.json` is used to store login-data and powermeter identification. The file `config.json.template` can be used as a blueprint.

To automate dual-recording analysis, the script has to log on to ZwiftPower and Garmin Connect. Login data are configured under the `garminconnect` and `zwiftpower` entries. Username and password are configured using their respective fields.

The script also tries to identify the power source used by analyzing the fit-files. This is what the `powermeters`-section in the configuration file is for. Each entry in the `powermeters`-section contains a name field and a list of matches that must be met in order to identify a powermeter. The `name`-field contains the name that should appear on ZwiftPower. Each key-value pair in the `matches`-section has to match entries in a `device_info`-section of the fit-file. All entries have to match in a single `device_info`-section for a positive identifcation to occur.

As far as I could tell fit-files from Zwift do not contain any information to identify connected devices. Hence, the powermeter can only be identified by finding the `manufacturer`-entry with the value `zwift` in a `device_info` structure. If a `manufacturer`-entry with value `zwift` is found, then the value from the `name`-entry is used for the fit-file from Zwift.

Fit-files from Garmin have more information. From the file `config.json.template` the script tries to match `manufacturer:garmin`, `garmin_product:2787` (Product ID for Garmin Vector 3` and `serial_number:3999631966` (redundant, as example only). You can analyze your fit-files using `fitdump`-utility from the `fitparse` python package or any other online fit-file analyzer to find matching criteria for your powermeter.

## TODO
* Add fallbacks if passwords are not found in config.json:
  - Keychain/password manager
  - Console promp
* Automate dual-recording upload for any ZwiftPower entry, not just the latest one
* Add another (fallback) backend to connect to Garmin Connect (for example gcexport).
