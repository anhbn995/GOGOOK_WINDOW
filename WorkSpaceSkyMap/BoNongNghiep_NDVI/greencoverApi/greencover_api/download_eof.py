from download_crop.download_from_eof import download_from_eof

def main(folderpath, input_url, token, list_month):
    workspace_path = download_from_eof(folderpath, input_url, token, list_month)
    return workspace_path

if __name__ == "__main__":
    list_month = None
    folderpath = '/home/nghipham/Desktop/Jupyter/data/DA/2_GreenSpaceSing'
    input_url = 'https://app.eofactory.ai/workspaces/0638e1a8-c4b4-4996-93b6-202ba4f23f6f/?menu=0'
    token = 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzI5OCwibmFtZSI6IlF1eWV0IE5ndXllbiBOaHUiLCJlbWFpbCI6InF1eWV0Lm5uQGVvZmFjdG9yeS5haSIsImNvdW50cnkiOiJWaWV0bmFtIiwicGljdHVyZSI6bnVsbCwiaWF0IjoxNjM2OTYzMjc1LCJleHAiOjE2Mzk1NTUyNzV9.IeMDOmepy7kpNT8BdiuH6ZEmMtPIC1PRyqsM80T-AhFDvYi4fGvUx096_PcYIuGcRIVrKYD6n2OfrU-WhB0h3Q'
    main(folderpath, input_url, token, list_month)