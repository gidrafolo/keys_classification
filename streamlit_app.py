
from pathlib import Path
import streamlit as st
from scripts.inference import model_predict

PROCESSED_DIR = "streamlit_data"
PATH_TO_MODEL = 'models\\KeysClassifier.pth'
def upload_data(out_dir: Path, text_on_button:str) -> None:
    """Upload multiple files.

    Args:
        out_dir (Path): dir to save files.
    """
    uploaded_audio = st.file_uploader(f"Drop here {text_on_button}", accept_multiple_files=True)
    for uploaded_file in uploaded_audio:
        out_dir.mkdir(exist_ok=True)
        save_uploadedfile(uploaded_file, out_dir)

    return uploaded_audio


def save_uploadedfile(uploadedfile, out_dir: Path) -> None:
    """Save files from buffer.

    Args:
        uploadedfile (UploadedFile): file's
        out_dir (Path): _description_
    """
    if uploadedfile.name not in out_dir.iterdir():
        (out_dir / uploadedfile.name).write_bytes(uploadedfile.getbuffer())
        st.success(f"Saved File:{uploadedfile.name} to data/streamlit/")



def pipeline_button(key_file: Path, data_dir: Path) -> None:
    """Process .dtu file to get model guess of key class.

    Args:
        out_dir (Path): dir to save the final file.
    """
    if st.button("Start pipeline"):
        key_file_name = key_file[0].name # get first file name
        st.write("In progress...")
        result = model_predict(PATH_TO_MODEL, str(data_dir / key_file_name))
        st.write(f"is key № {result}")

    else:
        st.write("Click to start pipeline")


if __name__ == "__main__":
    if "data_dir" not in st.session_state:
        data_dir = Path(PROCESSED_DIR)                   # / datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        st.session_state["data_dir"] = data_dir.as_posix()
    data_dir = Path(st.session_state["data_dir"])
    st.write(data_dir)
    key_file = upload_data(data_dir, text_on_button=".dtu file")
    pipeline_button(key_file, data_dir)
