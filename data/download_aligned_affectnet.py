import requests
import zipfile
import os

def download_file_from_google_drive(id, destination):

	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = {'id':id}, stream=True)
	token = get_confirm_token(response)

	if token:
		params = {'id': id, 'confirm':token}
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)

def get_confirm_token(response):

	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):

	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

if __name__ == "__main__":

	print ('Download AffectNet ... ')
	file_id = '1NOr_GLc5-P79MRMibJaVq6ckpEsHGhcn'
	destination = 'affectnet_182.zip'
	download_file_from_google_drive(file_id, destination)
	with zipfile.ZipFile(destination) as zf:
		zip_dir = zf.namelist()[0]
		zf.extractall('./')
	os.remove(destination)