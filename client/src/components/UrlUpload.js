import React from 'react';
import request from 'superagent';
import Loading from 'react-loading';

class UrlUpload extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      url: "",
      loading: false,
      formError: false
    }
  }

  render() {
    const onTextChange = (ev) => {
      this.setState({url: ev.target.value})
    }

    const requestStarted = () => {
      this.setState({loading: true, formError: false})
    }

    const onSubmit = (ev) => {
      requestStarted();

      ev.preventDefault();
      request.post('/citeomatic/upload-url')
        .send(JSON.stringify({
          'url': this.state.url,
      }))
      .set('Content-Type', 'application/json')
      .end((err, res) => {
        this.setState({loading: false});

        if (err) {
          this.setState({formError: true})
          window.mixpanel.track('Url upload pdf error');
        } else {
          this.props.onRequestFinished(res, 'url');
        }
      })
    }
    
    let errBox = null;
    if (this.state.formError) {
      errBox = (
      <div className="form-error">
        <p>There was an error with the URL you submitted. Please try a different URL or enter the data through the "Enter Paper Details" form.</p>
      </div>
    )};

    return (
      <div className="tab-panel">
       {errBox}
       <form method="GET" action="" onSubmit={onSubmit} data-toggle="validator" role="form">
          <div className="form-group">
            <label htmlFor="input-url">URL for an existing PDF</label>
            <input type="url" size={60} onChange={onTextChange} value={this.state.url} className="form-control" id="input-url" required />
            <p className="placeholder">Ex. https://pdfs.semanticscholar.org/e5ae/9c2093699913a480bc0b25c3cd3b958a6b18.pdf</p>
          </div>
          <input type="submit" value="Find Citations"/>
       </form>
        {this.state.loading && (<div className="loading-icon"><center><Loading type='spin' color='black'
        /></center></div>)}

      </div>
    );
  }
}

UrlUpload.propTypes = {
  onRequestFinished: React.PropTypes.func.isRequired
}

export default UrlUpload;
