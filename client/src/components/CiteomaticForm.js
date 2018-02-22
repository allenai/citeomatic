import React from 'react';
import request from 'superagent';
import Loading from 'react-loading';

class CiteomaticForm extends React.Component {
  constructor(props) {
    super(props);
    this.onSuccess = this.onSuccess.bind(this);

    this.onTitleChange = this.onTitleChange.bind(this);
    this.onAbstractChange = this.onAbstractChange.bind(this);
    this.onAuthorsChange = this.onAuthorsChange.bind(this);
    this.onSubmit = this.onSubmit.bind(this);

    this.state = {
      title: '',
      abstract: '',
      authors: '',
      loading: false,
      formError: false
    };
  }

  onSubmit(event) {
    event.preventDefault();
    if (this.state.title || this.state.abstract || this.state.authors) {
      console.log(event);
      this.setState({loading: true, formError: false});
      event.preventDefault();
      //Log that they submitted the form and what they tried

      request.post('/citeomatic/upload-form')
        .send(JSON.stringify({
          'title': this.state.title,
          'abstract': this.state.abstract,
          'authors': this.state.authors
        }))
        .set('Content-Type', 'application/json')
        .end(this.onSuccess);
    } else {
      this.setState({formError: "Please enter text for title, abstract or author."});
    }
  }

  onTitleChange(event) {
    this.setState({title: event.target.value});
  }

  onAbstractChange(event) {
    this.setState({abstract: event.target.value});
  }

  onAuthorsChange(event) {
    this.setState({authors: event.target.value});
  }

  onSuccess(err, res) {
    console.log('form posted!');
    console.log(err);
    console.log(res);
    console.log(this.props.onSuccess);
    if (err) {
      this.setState({formError: "There was an error submitting the form. Please change the input and try again.", loading: false})
      window.mixpanel.track('Paper form error');
    } else {
      this.props.onRequestFinished(res);
      this.setState({loading: false});
    }
  }

  render() {
    let errBox = null;
    if (this.state.formError) {
      errBox = <ErrorBox text={this.state.formError} />
    };

    return (
      <div className="tab-panel">
        {errBox}
        <form onSubmit={this.onSubmit}>
          <div className="form-group">
            <label htmlFor="paper_title">Title</label>
            <input onChange={this.onTitleChange} type="text" className="form-control" id="paper_title" name="title" placeholder="Deep Learning Overview" />
          </div>
          <div className="form-group">
            <label htmlFor="paper_abstract">Abstract</label>
            <textarea onChange={this.onAbstractChange} className="form-control" id="paper_abstract" name="abstract" rows="5" placeholder="In recent years, deep artificial neural networks (including recurrent ones) have won numerous contests in pattern recognition and machine learning..."></textarea>
          </div>
          <div className="form-group">
            <label htmlFor="paper_authors">Authors</label>
            <input onChange={this.onAuthorsChange} type="text" className="form-control" id="paper_authors" name="authors" placeholder="Christof Koch, Oren Etzioni" />
          </div>

          <div>
            <input type="submit" value="Find Citations"/>
          </div>
        </form>
        {this.state.loading && (<div className="loading-icon"><center><Loading type='spin' color='black'
        /></center></div>)}
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


CiteomaticForm.propTypes = {
  onRequestFinished: React.PropTypes.func.isRequired
}

export default CiteomaticForm;
