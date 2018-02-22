import React from 'react';
import Dropzone from 'react-dropzone';
import request from 'superagent';
import uploadPdfImage from '../images/upload@2x.png'

class PdfUpload extends React.Component {

  onDrop(acceptedFiles) {
    this.props.onRequestStarted();
    this.setState({
      files: acceptedFiles,
      formError: false
    });
    var req = request.post('/citeomatic/upload');
    acceptedFiles.forEach((file) => {
      req.attach(file.name, file);
    });
    req.end(this.onSuccess);
  }

  onSuccess(err, res) {
    if (err) {
      this.props.onLoadingFinished();
      this.setState({files: [], formError: "There was an error reading your PDF. Please try copy and pasting the contents into the \"Enter Paper Details\" form instead. (Our PDF parsing is still a work in progress. Thanks for your patience!)"})
      window.mixpanel.track(
        'PDF Upload failure'
      );

    } else {
      this.props.onRequestFinished(res);
    }
  }

  onOpenClick() {
    this.dropzone.open();
  }

  constructor(props) {
    super(props);
    this.onOpenClick = this.onOpenClick.bind(this);
    this.onDrop = this.onDrop.bind(this);
    this.onSuccess = this.onSuccess.bind(this);
    this.state = {
      files: [],
      formError: false,
      loading: false
    };
  }

  render() {
    let errBox = null;
    if (this.state.formError) {
      errBox = <ErrorBox text={this.state.formError} />
    };

    return (
      <div>
        {errBox}
        <Dropzone ref={(node) => { this.dropzone = node; }} onDrop={this.onDrop} className="dropzone" accept="application/pdf" minSize={1000} maxSize={10000000} multiple={false}>
        <div className="centered column">
          <img className="upload-img" alt="pdf upload" src={uploadPdfImage}/>
            <p className="browse-pdf-text">Drag and drop your pdf here or <a>browse</a></p>
        </div>
        </Dropzone>
        {this.state.files.length > 0 ? <div>
        <h2>Uploading {this.state.files.length} file...</h2>
        </div> : null}
      </div>
    );
  }
}

function ErrorBox(props) {
  return (
    <div className="form-error">
      <p>{props.text}</p>
    </div>
  )
}


PdfUpload.propTypes = {
  onRequestFinished: React.PropTypes.func.isRequired,
  onRequestStarted: React.PropTypes.func.isRequired,
  onLoadingFinished: React.PropTypes.func.isRequired
}

export default PdfUpload;
