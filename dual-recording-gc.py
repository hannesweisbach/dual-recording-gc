#!/usr/bin/env python3

import argparse
import collections
import datetime
import functools
import io
import json
import os
import re
import subprocess
import tempfile
import typing

import fitparse

from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectAuthenticationError,
)
import zipfile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import requests
from lxml import html
import cloudscraper
from urllib.parse import urlparse, parse_qsl

import logging
logging.basicConfig(level=logging.WARNING)


def dates_close(t1: datetime.datetime,
                t2: datetime.datetime,
                td: datetime.timedelta = datetime.timedelta(minutes = 30)):
  return (t1 - td < t2) and (t1 + td > t2)

class Config(object):
  _ConfigFile: typing.Optional[str] = None
  _Config: typing.Optional[dict] = None

  def __init__(self, config_file):

    assert os.path.exists(config_file)
    
    if Config._ConfigFile != None:
      raise RuntimeError(f'Config file {Config._ConfigFile} already loaded.')

    Config._ConfigFile = config_file
    with open(config_file, 'r') as f:
      Config._Config = json.load(f)

  @staticmethod
  def _credentials(which: str) -> (str, str):
    return (Config._Config[which]['username'], Config._Config[which]['password'])

  @staticmethod
  def zwiftpower_credentials() -> (str, str):
    return Config._credentials('zwiftpower')

  @staticmethod
  def garminconnect_credentials() -> (str, str):
    return Config._credentials('garminconnect')

  @staticmethod
  def powermeter_names():
    return Config._Config['powermeters']

# Garmin timestamp offset -631065600

def powermeter_name(fitfile, powermeters) -> str:
  def in_record(name: str, value: str, records):
    for record in records:
      if record.name == name and str(record.value) == value:
        return True
    return False
  def all_in_record(dict_ : dict, records):
    for name, value in dict_.items():
      if not in_record(name, value, records):
        return False
    return True

  devices = fitfile.get_messages('device_info')
  for records in devices:
    for powermeter in powermeters:
      if all_in_record(powermeter['matches'], records):
        return powermeter['name']
  
  raise RuntimeError("Could not identify power meter")

class ZwiftPower:
  Event = collections.namedtuple('Event', ['date', 'title', 'id'])
  _NoneEvent = Event(date = None, title = "", id = "")

  class Activity:
    def __init__(self, html_option, events):
      self._url = html_option.attrib.get('value')
      self._title_name = html_option.attrib.get('title_name')
      datestr, self._name = [part.strip() for part in html_option.text.split(' : ', 1)]
      self._date = datetime.datetime.fromisoformat(datestr.replace('/', '-'))

      event_candidates = list(filter(lambda evt : dates_close(self.date, evt.date), events))

      if len(event_candidates):
        print(f'Event candidates for activity "{self.title_name}" @{self.date}')
        for e in event_candidates:
          print(f'  "{e.title}" @{e.date}')

      self._event = next(iter(event_candidates), ZwiftPower._NoneEvent)

    @property
    @functools.cache
    def fitfile(self):
      response = requests.get(self.url)
      response.raise_for_status()
      return fitparse.FitFile(response.content)

    @property
    def url(self):
      return self._url

    @property
    def title_name(self):
      return self._title_name

    @property
    def event_name(self):
      return self._event.title

    @property
    def event_id(self):
      return self._event.id

    @property
    def date(self):
      return self._date

    def __str__(self):
      return f'{self._date} {self._name} {self._url}'

  def __init__(self, username, password):
    self.session = cloudscraper.CloudScraper()
    #self.activities = None
    payload = {
      'username' : username,
      'password' : password,
      'login' : 'Login'
    }
    r = self.session.post('https://zwiftpower.com/ucp.php', params = { 'mode': 'login'}, data = payload)
    
    zid_match = re.search(r"zid : '(?P<zid>(\d*))'", r.text)
    if not zid_match:
      raise RuntimeError("Could not extract Zwift ID")
    self.zid = zid_match.group('zid')

  def update_activities(self):
    Zwiftpower.activities.fget.cache_clear()
    return self.activities

  @property
  @functools.cache
  def activities(self):
    r = self.session.get('https://zwiftpower.com/analysis.php')
    #print(r.text)
    tree = html.fromstring(r.text)
    html_events = tree.xpath('//div[@id="tab_main"]//select[@name="set_zwift_event_id"]/option')[1:]
    def create_event(html_event):
        datestr, raw_name = html_event.text.split(' - ', maxsplit = 1)
        date = datetime.datetime.fromisoformat(datestr.replace('/', '-'))
        name = raw_name.split(' [')[0]
        id_ = html_event.attrib.get('value')
        return ZwiftPower.Event(date, name, id_)

    events = [create_event(e) for e in html_events]

    print('Found these events:')
    for e in events:
      print(f'  {e.date} {e.title}')

    options = tree.xpath('//div[@id="tab_main"]//select[@name="zwift_activity_filename"]/option')
    return [ZwiftPower.Activity(option, events) for option in options[1:]]

  def select_activity(self):
    for i, act in enumerate(self.activities):
      print(f"{i} {act}")

    while True:
      try:
        idx = int(input('Select activity: '))
      except ValueError:
        print("Could not convert data to an integer.")
      else:
        return idx

  @property
  @functools.cache
  def list(self):
    # Already submitted dual-power events
    # https://zwiftpower.com/api3.php?do=analysis_list&zwift_id=4023691&_=1641060125610
    params = {
      'do': 'analysis_list',
      'zwift_id': self.zid,
    }
    r = self.session.post('https://zwiftpower.com/api3.php', params = params)
    self.analysis_list = json.loads(r.text)['data']

    for e in self.analysis_list:
      e['date'] = datetime.datetime.fromtimestamp(int(e['date']))

    print("Activities that already have dual recording data:")
    for e in self.analysis_list:
      print(f"  {e['date']} {e['title']}")

    return self.analysis_list

  def download_activity_interactive(self):
    idx = self.select_activity()
    return self.activities[idx]

  def upload_secondary_power_source(self, activity, fitfiles, public = True, set_zwift_event = True):
    num_files = len(fitfiles)
    MAX_FILES = 3
    pmeters = Config.powermeter_names()
    pmeter_name = powermeter_name(activity.fitfile, pmeters)
    data = {
      'set_name' : activity.title_name,
      'zwift_activity_filename': activity.url,
      'activityname' : pmeter_name,
      'set_zwift_event_id': activity.event_name if set_zwift_event else '0',
      'set_public': '1' if public else '0',
    }

    files = {}
    for i in range(MAX_FILES):
      data[f'file{i+1}name'] = powermeter_name(fitparse.FitFile(fitfiles[i][0]), pmeters) if i < num_files else ''
      files[f'file{i+1}'] = open(fitfiles[i][0], 'rb') if i < num_files else ('', '')

    params = { 'do': 'upload'}
    r = self.session.post('https://zwiftpower.com/analysis.php', params = params, data = data, files = files)

    #print(r.text)

    params = dict(parse_qsl(urlparse(r.url).query))
    params['do'] = 'save_graph_settings'

    file_ids = [fid.attrib.get('value') for fid in html.fromstring(r.content).xpath('//div[@id="tab_settings"]//input[@type="checkbox"]')]

    d = {}

    for i in range(MAX_FILES + 1):
      d[f'file_id_{i}'] = file_ids[i] if i < len(file_ids) else ""
      d[f'data_filename_{i}'] = pmeter_name if i == 0 else data[f'file{i}name']
      d[f'timestamp_offset_{i}'] = fitfiles[i-1][1] if i > 0 and i-1 < num_files else "0"
      d[f'timestamp_start_{i}'] = "0"
      d[f'timestamp_end_{i}'] = "0"
    print(d)

    # post analysis.php
    #params = { 'do': 'save_graph_settings', 'set_id' : }

    r = self.session.post('https://zwiftpower.com/analysis.php', params = params, data = d)
    print(r)


  def remove_dataset(self, setid):
    params = { 'do': 'remove_set', 'set_id' : f'{setid}'}
    self.session.post('https://zwiftpower.com/analysis.php', params = params)

def garmin_retry(func, text):
  while True:
    try:
      func()
    except GarminConnectConnectionError as e:
      print(text, e)
    else:
      break

def download_gc_connect(date):
  #gcexport = '/tmp/garmin-connect-export/gcexport.py'
  #args = [
  #  gcexport,
  #  '--username', 'hannes.weisbach@gmail.com',
  #  '--password', 'foobarGarmin1',
  #  '--unzip',
  #  '--format', 'original',
  #  '--start_activity_no', f'{idx}',
  #  '--count', '1'
  #]
  #subprocess.run(args
  #retryer = Retryer(
  #  delay_strategy=ExponentialBackoffDelayStrategy(initial_delay=timedelta(seconds=1)),
  #  stop_strategy=MaxRetriesStopStrategy(5))

  gc_user, gc_pass = Config.garminconnect_credentials()
  client = Garmin(gc_user, gc_pass)
  garmin_retry(lambda: client.login(), "Login failed. Retrying …")
  start = date.strftime('%Y-%m-%d')
  #end = (date + datetime.timedelta(days = 1)).strftime('%Y-%m-%d')
  #print(start, end)
  actlist = client.get_activities_by_date(start, start, activitytype = 'cycling')
  for act in actlist:
    #print(act)
    print(act['activityId'], act['startTimeLocal'])

  if len(actlist) == 0:
    raise RuntimeError("No activities found on Garmin Connect. Aborting.")

  # TODO: try matching by time too
  actid = actlist[0]['activityId']
  print(f"Download activity {actid}")
  zipped = client.download_activity(actid, Garmin.ActivityDownloadFormat.ORIGINAL)
  z = zipfile.ZipFile(io.BytesIO(zipped))
  print(z.infolist())

  # there seems to be only the fit file in the zip archive.
  targetdir = tempfile.mkdtemp()
  fitfile = z.extract(z.infolist()[0], path = targetdir)
  print(fitfile)

  return fitfile

# fill "holes" in fit file due to 'smart recording'. This is required to align power curves.
def fixup_fit_file(arr):
  time_begin = arr[0, 0]
  time_end = arr[-1, 0]
  duration = time_end - time_begin
  entries = arr.shape[0]


  if (entries - 1) != duration:
    print(f"Fixup enabled and required. Duration {duration}, Entries {entries}")
    adjacent_differences = np.diff(arr[:,0])
    # filter out where timestamps are not 1s apart
    indices = np.nonzero(adjacent_differences != 1)[0]
    timestamps = arr[indices][:, 0]
    gap_size = adjacent_differences[indices]
    offset = 0
    for i, ts, gap in zip(indices, timestamps, gap_size):
      adjusted_index = i + offset
      # generate fill values: incresing timestamps, but keep the power value
      values = np.array([arr[adjusted_index] + [seconds, 0] for seconds in range(1, gap)])
      #print(f"Hole @{i}: {gap} {values.shape} offset: {offset}")
      arr = np.insert(arr, obj = 1 + adjusted_index, values = values, axis = 0)
      #arr grew - adjust indices of future infills
      offset = offset + values.shape[0]
  else:
    print(f"Fixup not required.")

  return arr

def read_power_from_fit_file(infile, outfile = None, fixup = False):
  fitfile = infile if isinstance(infile, fitparse.FitFile) else fitparse.FitFile(infile)
  fitfile_fields = ((record.get_raw_value('timestamp'), record.get_value('power')) for record in fitfile.get_messages('record'))
  if outfile:
    with open(outfile, 'w') as output:
        for line in fitfile_fields:
          output.write(f"{line[0]} {line[1]}")
    return outfile
  else:
    arr = np.array([line for line in fitfile_fields], dtype = (int, int))
    if fixup:
      arr = fixup_fit_file(arr)
    return arr

def ssd(v1, v2):
  # sum of squared differences
  diff = v1 - v2
  return np.dot(diff, diff)

def find_min_overlap(fixed, moving):
  mlen = moving.shape[0]
  flen = fixed.shape[0]
  f_length = 2 * mlen + flen

  fixedvector = np.zeros(f_length, dtype = int)
  movingvector = np.zeros(2 * (flen + mlen) + mlen, dtype = int)

  fixedvector[mlen:mlen + flen] = fixed
  movingvector[flen+mlen:flen+mlen+mlen] = moving
  min_ = np.dot(fixedvector, fixedvector) + np.dot(movingvector, movingvector)
  shift = 0

  for i in range(mlen + flen):
    m_start = mlen + flen - i
    ssd_ = ssd(fixedvector, movingvector[m_start:m_start + f_length])
    if ssd_ < min_:
      #print(ssd_, min_, shift)
      min_ = ssd_
      shift = i

  print(shift, min_, mlen, flen)

  return shift - mlen

def syncronize_files(fixed, moving, fixup = True, plot = False):
  power_fixed = read_power_from_fit_file(fixed, fixup = fixup)
  power_moving = read_power_from_fit_file(moving, fixup = fixup)

  dynamic_shift = find_min_overlap(fixed = power_fixed[:,1], moving = power_moving[:,1])

  static_shift = power_fixed[0, 0] - power_moving[0, 0]
  shift = static_shift + dynamic_shift

  print(f"Shift between start timestamps of files: {static_shift}")
  print(f"Shift to matches power curves closest: {dynamic_shift}")
  print(f"Total shift: {shift}")

  if plot:
    plt.plot(range(0, power_fixed.shape[0]), power_fixed[:,1], label = "Direto")
    plt.plot(dynamic_shift + np.array(range(0, power_moving.shape[0])), power_moving[:,1], label = "Garmin")
    plt.legend()
    plt.show()

  return shift

def upload_dual_record_latest(dryrun = False, plot = False):
  zp_user, zp_pass = Config.zwiftpower_credentials()
  zp = ZwiftPower(zp_user, zp_pass)
  dual_recorded = zp.list
  activity = zp.activities[0]

  # match by name + check if dates are 'close'
  already_uploaded = [dr for dr in dual_recorded if activity.title_name == dr['title'] and dates_close(activity.date, dr['date'])]
  if len(already_uploaded):
    print(f"Newest activity already dual-powered:")
    print(f"activity:   '{activity.date} {activity.title_name}'")
    for e in already_uploaded:
      print(f"dual-power: '{e['date']} {e['title']}")
    exit(1)
 
  garmin_fitfile = download_gc_connect(activity.date)

  timediff = syncronize_files(fixed = activity.fitfile, moving = garmin_fitfile, fixup=True, plot = plot)

  if not dryrun:
    zp.upload_secondary_power_source(activity, [(garmin_fitfile, timediff)])


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--latest')

  #args = parser.parse_args()

  Config('config.json')

  upload_dual_record_latest()
  exit(0)

  zp_user, zp_pass = Config.zwiftpower_credentials()
  zp = ZwiftPower(zp_user, zp_pass)
  activity = zp.download_activity_interactive()

