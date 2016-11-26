from selenium import webdriver
from selenium.webdriver.support import ui
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
#import webbrowser
import time
import pyperclip
#import os

driver = webdriver.Chrome()
passCode = ''
userName = ''
import ctypes
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( hexKeyCode, 0x48, 0, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( hexKeyCode, 0x48, 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))   
    
## do i need this?
def page_is_loaded(driver):
    return driver.find_element_by_tag_name("body") != None
    
def PressAltTab():
    PressKey(0x012) #Alt
    PressKey(0x09) #Tab
    ReleaseKey(0x09) #~Tab
    ReleaseKey(0x012) #~Alt
def NewTab():
    PressKey(0x011) #Ctrl
    PressKey(0x54) #T
    ReleaseKey(0x54)
    ReleaseKey(0x011)
    
def DuplicateTab(): # not working when I wanted it to. 
    PressKey(0x012) #Alt
    PressKey(0x44) #d
    ReleaseKey(0x44) #~d
    PressKey(0x0D) #enter
    ReleaseKey(0x0D) #~enter
    ReleaseKey(0x012) #~Alt

def SaveAs():
    saveas = ActionChains(driver).key_down(Keys.CONTROL).send_keys('s').key_up(Keys.CONTROL)
    saveas.perform()
def Enter():
    PressKey(0x0D)
    time.sleep(0.5)
    ReleaseKey(0x0D)
def Save():
    PressKey(0x11) #ctrl
    PressKey(0x53) #S
    ReleaseKey(0x53) 
    ReleaseKey(0x11)
def Paste():
    PressKey(0x11) #ctrl
    PressKey(0x56) #V
    ReleaseKey(0x56) 
    ReleaseKey(0x11)
def CopyTitle():
    sum_title = driver.find_element_by_tag_name('h1').text.encode('ascii','ignore')
    title = sum_title.replace(":","-").replace(" ","_")
    time.sleep(1)
    pyperclip.copy(title)
def F6():
    PressKey(0x75)
    ReleaseKey(0x75)
    
def dl_mov(url):
    pyperclip.copy(url)
    time.sleep(1)
    F6()
    time.sleep(.5)
    Paste()
    time.sleep(.5)
    Enter()
    title = mov_title[u].replace(":","-").replace(" ","_")
    pyperclip.copy(title)    
    time.sleep(2)
    Save()
    time.sleep(1)
    Paste()
    time.sleep(.5)
    Enter()
    time.sleep(3)
def dl_html(url):
    tab = driver.get(url)
    wait = ui.WebDriverWait(driver, 10)
    wait.until(page_is_loaded)
    time.sleep(2)
    CopyTitle()
    SaveAs()
    time.sleep(1)
    Paste()
    time.sleep(1)
    Enter()
    
def dl_text(url):
    tab = driver.get(url)
    wait = ui.WebDriverWait(driver, 10)
    wait.until(page_is_loaded)
    time.sleep(2)
    pyperclip.copy(mov_t[t])
    Save()
    time.sleep(1)
    Paste()
    time.sleep(1)
    Enter()
       

driver.get("https://support.sas.com/edu/viewmyelearn.html")
wait = ui.WebDriverWait(driver, 10)
wait.until(page_is_loaded)
email_field = driver.find_element_by_id("IDToken1")
email_field.send_keys(userName + "@ncsu.edu")
password_field = driver.find_element_by_id("IDToken2")
password_field.send_keys(passCode)
password_field.send_keys(Keys.RETURN)
wait.until(page_is_loaded)
elem = driver.find_element_by_link_text("Applied Analytics Using SAS Enterprise Miner (EM 5.3 up to 13.2) (retiring April 2017)")
elem.click()
time.sleep(1)

elem = driver.find_element_by_link_text("Learning Path")
elem.click()
time.sleep(1)

titleOfLesson = "Introduction to Using SAS Enterprise Miner"
elem = driver.find_element_by_link_text(titleOfLesson)
elem.click()
time.sleep(1)



#==============================================================================
# driver.quit()
#==============================================================================

#PressKey(0x09) #Tab
#ReleaseKey(0x09) #~Tab
#PressKey(0x0D) #Enter
#ReleaseKey(0x0D) #~Enter
#time.sleep(8)

#PressAltTab()
#time.sleep(2)
#F6()
#pyperclip.copy(mov_text_urls[0])
#Paste()
#Enter()



#tab = driver.find_element_by_tag_name("body")
#tab.send_keys(Keys.CONTROL + 't')
#time.sleep(2)
#F6()
#F6()
#Paste()
#Enter()
#time.sleep(20)


#==============================================================================
# time.sleep(5)
# for t in range(0,len(mov_text_urls)):
#     print(t)
#     dl_text(mov_text_urls[t])
#     
#==============================================================================

#for u in range(1,len(mov_urls)):
#    dl_mov(mov_urls[u])
    
#for s in summary_urls:
#    dl_html(s)

#for q in quiz_urls:
#    dl_html(q)


# https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/keyCode
####################################

#old
#def dl_mov(url):
#    tab = driver.get(url)
#    wait = ui.WebDriverWait(driver, 10)
#    wait.until(page_is_loaded)
#    time.sleep(4)
#    SaveAs()
#    time.sleep(4)
#    Enter()
#    time.sleep(8)