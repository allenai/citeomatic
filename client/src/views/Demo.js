import React from 'react';
import Results from './Results';
import request from 'superagent';

export default class Demo extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      papers: []
    };
  }

  componentDidMount() {
    const onSuccess = (res) => {
      this.setState({
        papers: res.body.predictions,
        source_file: res.body.source_file,
        title: res.body.title,
        abstract: res.body.abstract,
        authors: res.body.authors,
        year: res.body.year
      })
    };

    const url = this.props.location.query.url;
    request.post('/citeomatic/upload-url')
      .send(JSON.stringify({ url }))
      .set('Content-Type', 'application/json')
      .end((err, res) => {
        onSuccess(res);
      })
  }

  render() {
    const papers = this.state.papers;
    if (papers && papers.length > 0) {
      return <Results
        onRestart={this.onRestart}
        isFetching={false}
        papers={papers}
        sourceFile={this.state.source_file}
        title={this.state.title}
        year={this.state.year}
        abstract={this.state.abstract}
        authors={this.state.authors} />
    } else {
      return <div>Loading...</div>
    }
  }
}
Demo.contextTypes = {
  router: React.PropTypes.object.isRequired
}
