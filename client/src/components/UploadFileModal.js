import React, { PropTypes } from 'react';

import PdfUpload from './PdfUpload';

import Loading from 'react-loading';

class UploadFileModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: false
    };
  }

  render() {

  	const requestStarted = () => {
      this.setState({loading: true})
  	}

    const loadingFinished = () => {
      this.setState({loading: false})
    }

  	const requestFinished = (response) => {
      this.setState({loading: false})
      this.props.onRequestFinished(response, 'pdf')
		}

    return (
      <div className="tab-panel">
        <div className="file-upload">
          <PdfUpload onLoadingFinished={loadingFinished} onRequestFinished={requestFinished} onRequestStarted={requestStarted}/>
        </div>
        {this.state.loading && (<div className="loading-icon"><center><Loading type='spin' color='black'
        /></center></div>)}
      </div>
    );
  }
}

UploadFileModal.propTypes = {
  onRequestFinished: PropTypes.func.isRequired
}

export default UploadFileModal;
