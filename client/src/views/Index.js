import React from 'react';

import { browserHistory, Link } from 'react-router';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';

import UploadFileModal from '../components/UploadFileModal';
import Results from './Results';
import CiteomaticForm from '../components/CiteomaticForm';
import UrlUpload from '../components/UrlUpload';

import citeomaticQuoteImage from '../images/citeomatic@2x.png';

class Index extends React.Component {
  constructor(props) {
    super(props);
    this.onPdfUploadFinished = this.onPdfUploadFinished.bind(this);
    this.onUrlUploadFinished = this.onUrlUploadFinished.bind(this);
    this.onFormPost = this.onFormPost.bind(this);
    this.setStateFromResponse = this.setStateFromResponse.bind(this);
    this.state = {
      papers: []
    };
  }

  setStateFromResponse(res) {
    this.setState({
      papers: res.body.predictions,
      source_file: res.body.source_file,
      title: res.body.title,
      abstract: res.body.abstract,
      authors: res.body.authors,
      year: res.body.year,
      cache_key: res.body.cache_key
    });
  }

  onUrlUploadFinished(res) {
    this.setStateFromResponse(res);

    //Log they successfully got a response and log number of citations returned
    window.mixpanel.track(
      'Url upload pdf success',
      {
        'papers_returned': res.body.predictions.length,
        'source_file': res.body.source_file,
        'title': res.body.title,
        'abstract': res.body.abstract,
        'authors': res.body.authors,
        'year': res.body.year,
        'num_uncited_papers' : res.body.predictions.filter(function (paper) { return paper.cited === ""; }).length,
        'num_cited_papers' : res.body.predictions.filter(function (paper) { return paper.cited !== ""; }).length
      }
    );

    browserHistory.push('/citeomatic/url/' + this.state.cache_key);
  }

  onPdfUploadFinished(res) {
    this.setStateFromResponse(res);

    //Log they successfully got a response and log number of citations returned
    window.mixpanel.track(
      'Uploaded pdf success',
      {
        'papers_returned': res.body.predictions.length,
        'source_file': res.body.source_file,
        'title': res.body.title,
        'abstract': res.body.abstract,
        'authors': res.body.authors,
        'year': res.body.year,
        'num_uncited_papers' : res.body.predictions.filter(function (paper) { return paper.cited === ""; }).length,
        'num_cited_papers' : res.body.predictions.filter(function (paper) { return paper.cited !== ""; }).length
      }
    );

    browserHistory.push('/citeomatic/pdf/' + this.state.cache_key)
  }

  onFormPost(res) {
    this.setStateFromResponse(res);
    //Log they successfully used the form to get a citeomatic response
    window.mixpanel.track(
      'Paper form success',
      {
        'papers_returned': res.body.predictions.length,
        'source_file': res.body.source_file,
        'title': res.body.title,
        'abstract': res.body.abstract,
        'authors': res.body.authors,
        'year': res.body.year
      }
    );

    browserHistory.push('/citeomatic/form/' + this.state.cache_key);
  }

  handleSelect(index, last) {
    //console.log('Selected tab: ' + index + ', Last tab: ' + last);
  }

  render() {
    let view = null;
    let papers = this.state.papers;

    if (papers && papers.length > 0) {
      window.mixpanel.track('Render results from input');
      view = <Results
        isFetching={false}
        papers={papers}
        sourceFile={this.state.source_file}
        title={this.state.title}
        year={this.state.year}
        abstract={this.state.abstract}
        authors={this.state.authors} />
    } else {
      window.mixpanel.track('Render homepage');
      view = (
        <div>
          <h3 className="intro">
            <img alt="citeomatic-quote-logo" src={citeomaticQuoteImage} className="quote-logo"/>
            Citeomatic
          </h3>
          <p className="tagline">
            Citeomatic finds new citations for you.
          </p>
          <p className="intro">
            Afraid of missing out on important references? Give us details about your paper and we'll automatically recommend papers you might want to cite.
            <br/>
          </p>
        <div className="centered">
          <div className="input-tabs">
            <Tabs
              onSelect={this.handleSelect}
              selectedIndex={0}
            >
              <TabList>
                <Tab>Upload PDF</Tab>
                <Tab>Input URL</Tab>
                <Tab>Enter Paper Details</Tab>
              </TabList>
              <TabPanel>
                <UploadFileModal onRequestFinished={this.onPdfUploadFinished} />
              </TabPanel>
              <TabPanel>
                <UrlUpload onRequestFinished={this.onUrlUploadFinished} />
              </TabPanel>
              <TabPanel>
                <CiteomaticForm onRequestFinished={this.onFormPost} />
              </TabPanel>
            </Tabs>
          </div>
        </div>
          <p className="disclaimer"><b>Disclaimer</b></p>
          <div className="notes centered">
            <ul>
              <li>Citeomatic is an early prototype -- please expect and report bugs!</li>
              <li><Link to="/citeomatic/privacy">We don't save the content of your paper</Link></li>
              <li>Citeomatic is limited to the Semantic Scholar corpus.  Papers from outside Computer Science
                  or Neuroscience will not give good results.</li>
            </ul>
          </div>
      </div>
      );
    }

    return view;
  }
}
export default Index;
