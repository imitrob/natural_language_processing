
import os
path = os.path.dirname(os.path.abspath(__file__))
package_path = "/".join(path.split("/")[:-1])
tts_path = path+"/text_to_speech"