import React from 'react';
import { browserHistory } from 'react-router';
import Modal from 'react-modal';

import YourPaper from '../components/YourPaper.js'
import CitationResults from '../components/CitationResults.js'

import quoteImage from '../images/cited.svg'

import request from 'superagent'

class Results extends React.Component {
  constructor(props) {
    super(props);
    this.downloadAllPapersClicked = this.downloadAllPapersClicked.bind(this);
    this.exportAllCitationsClicked = this.exportAllCitationsClicked.bind(this);
    this.onRestart = this.onRestart.bind(this);
    this.onCacheFetch = this.onCacheFetch.bind(this);
    this.setStateFromResponse = this.setStateFromResponse.bind(this);

    this.state = {
      papers: props.papers,
      abstract: props.abstract,
      title: props.title,
      year: props.year,
      authors: props.authors,
      sourceFile: props.sourceFile,
      showCombinedBibtexModal: false,
      uncitedPapers : props.papers && props.papers.filter(function (paper) { return paper.cited === ""; }),
      citedPapers : props.papers && props.papers.filter(function (paper) { return paper.cited !== ""; })
    }
  }

  componentWillMount() {
    let cacheKey = this.props.params && this.props.params.cacheKey;
    if (cacheKey) {
      let check_path = '';
      let path = this.props.route.path;

      if (path.startsWith('pdf/')) {
        check_path = 'pdf'
      } else if (path.startsWith('url/')) {
        check_path = 'url'
      } else if (path.startsWith('form/')) {
        check_path = 'form'
      }

      request.get('/citeomatic/cache_check/' + check_path + '/' + cacheKey)
        .set('Content-Type', 'application/json')
        .end(this.onCacheFetch);
    }
  }

  onCacheFetch(err, res) {
    if (res.body == null) {
      this.onRestart();
      return;
    } else {
      window.mixpanel.track('Render results from cache');
      this.setStateFromResponse(res);
    }
  }

  setStateFromResponse(res) {
    this.setState({
      papers: res.body.predictions,
      source_file: res.body.source_file,
      title: res.body.title,
      abstract: res.body.abstract,
      authors: res.body.authors,
      year: res.body.year,
      cache_key: res.body.cache_key,
      showCombinedBibtexModal: false,
      uncitedPapers : res.body.predictions.filter(function (paper) { return paper.cited === ""; }),
      citedPapers : res.body.predictions.filter(function (paper) { return paper.cited !== ""; })
    });
  }

  onRestart() {
    browserHistory.push('/citeomatic');
  }


  downloadAllPapersClicked() {
    window.mixpanel.track('DownloadAllPapers clicked');
    const ids = this.state.papers.map(p => p.document.id).join(",");
    window.open(`/citeomatic/api/pdfs?ids=${ids}`, '_blank');
  }

  exportAllCitationsClicked() {
    window.mixpanel.track('ExportAllCitations clicked');
    this.setState({
      showCombinedBibtexModal: true
    })
  }

  render() {
    const bibtex = this.state.papers && this.state.papers.map(p => p.bibtex).join('\n');
    const papers = this.state.papers;
    if (papers && papers.length > 0) {
      const citedPapersText = this.state.citedPapers.length > 0 ? <span>We found <a href="#new_citations"><b>{this.state.uncitedPapers.length}</b></a> new citations and <a href="#old_citations"><b>{this.state.citedPapers.length}</b></a> that you have already cited.</span> : <span>We found <a href="#new_citations"><b>{this.state.uncitedPapers.length}</b></a> new citations.</span>;
      return (
        <div>
          <div className="container nav">
            <div className="justify-left back-nav">
              <p className="link-home"><a onClick={this.onRestart} >&lt; Run Citeomatic on a different paper</a></p>
            </div>
            <div className="justify-right">
              <p className="link-home">Citeomatic</p>
            </div>
          </div>
          <YourPaper
            sourceFile={this.state.sourceFile}
            title={this.state.title}
            year={this.state.year}
            abstract={this.state.abstract}
            authors={this.state.authors}
            numCitations={this.state.papers.length} />
          <div className="container actions">
              <h3 className="result-header">
                {citedPapersText}
              </h3>
              <div className="downloads justify-right">
                <a onClick={this.exportAllCitationsClicked} className="export-citations btn btn-default">
                  <img alt="download citation" src={quoteImage} height="20" width="20" className="download-actions" />
                  Export Citations
                </a>
              </div>
          </div>
            <div className="cited-results">
              <CitationResults title="Current Citations" anchor="old_citations" papers={this.state.citedPapers} />
            </div>
            <div className="uncited-results">
              <CitationResults title="New Citations" anchor="new_citations" papers={this.state.uncitedPapers} />
            </div>
          <Modal
              className="Modal__Bootstrap modal-dialog"
              portalClassName="modal-portal"
              closeTimeoutMS={10}
              isOpen={this.state.showCombinedBibtexModal}
              contentLabel=""
              onRequestClose={() => this.setState({showCombinedBibtexModal: false})}>
              <pre>
                <h3>All citations for {this.state.title}</h3>
                {bibtex}
              </pre>
          </Modal>
        </div>
      );
    } else {
      return (<div>Loading...</div>);
    }
  }
}

export default Results;
