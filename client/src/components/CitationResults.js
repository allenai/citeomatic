import React from 'react';
import Modal from 'react-modal';

import pdfImage from '../images/pdf.svg'
import quoteImage from '../images/cited.svg'

class CitationResults extends React.Component {
  constructor(props) {
    super(props);
    this.notRelevantClicked = this.notRelevantClicked.bind(this);
    this.downloadPdfClicked = this.downloadPdfClicked.bind(this);

    this.state = {
      bibTexModal: null
    }
  }

  notRelevantClicked() {
    console.log("Hooray not relevant was clicked");
  }

  downloadPdfClicked(doc) {
    window.mixpanel.track('Download PDF clicked', 
      {
        'docId': doc.id,
        'authors': doc.authors,
        'title': doc.title,
        'year': doc.year
      }
    );
  }

  openBibtexModal(doc) {
    this.setState({ bibTexModal: doc.id });
    window.mixpanel.track('OpenBibtex clicked', 
      {
        'docId': doc.id,
        'authors': doc.authors,
        'title': doc.title,
        'year': doc.year
      }
    );
  }

  closeBibtexModal() {
    this.setState({ bibTexModal: null });
  }

  paperTitleClicked(doc) {
    window.mixpanel.track('Result title clicked',
      {
        'docId': doc.id,
        'authors': doc.authors,
        'title': doc.title,
        'year': doc.year
      }
    );
  }

  render() {
    const renderDocument = (doc) => {
      const s2Url = 'http://www.semanticscholar.org/paper/' + doc.document.id;
      return (
      <tr key={doc.document.id}>
        <td><a onClick={() => this.paperTitleClicked(doc.document)} href={s2Url} target="_blank">{doc.document.title}</a></td>
        <td className="author-column">{doc.document.authors.join(', ')}</td>
        <td>{doc.document.year}</td>
        <td>
          <div className="progress">
            <div className="progress-bar" role="progressbar" aria-valuenow="70" aria-valuemin="0" aria-valuemax="100" style={{"width": (100 * doc.score) + "%"}}>
              <span className="sr-only">{doc.score}% Complete</span>
            </div>
          </div>
        </td>
        <td>
        <a onClick={() => this.downloadPdfClicked(doc.document)} href={doc.pdf} target="_blank"><img src={pdfImage} height="25" width="25" alt="pdf" /></a></td>
        <td>
          <a onClick={() => this.openBibtexModal(doc.document)} >
            <img src={quoteImage} height="25" width="25" alt="quote" />
          </a>
          <Modal
            className="Modal__Bootstrap modal-dialog"
            portalClassName="modal-portal"
            closeTimeoutMS={10}
            isOpen={this.state.bibTexModal === doc.document.id}
            contentLabel=""
            onRequestClose={() => this.closeBibtexModal(doc.document.id)}>
            <pre>{doc.bibtex}</pre>
          </Modal>
        </td>
      </tr>
    )};

    const listItems = this.props.papers.map(renderDocument);
    if (listItems.length <= 0) { return null };
    return (
      <div className="container citation-table">
        <h4 id={this.props.anchor}>{this.props.title}</h4>
        <table className="table table-striped">
          <thead>
            <tr>
              <th>Title</th>
              <th>Authors</th>
              <th>Year</th>
              <th>Confidence</th>
              <th>PDF</th>
              <th>Cite</th>
            </tr>
          </thead>
          <tbody>
            {listItems}
          </tbody>
        </table>
      </div>
    );
  }
}

CitationResults.propTypes = {
  papers: React.PropTypes.array.isRequired,
  title: React.PropTypes.string.isRequired,
  anchor: React.PropTypes.string.isRequired
}

export default CitationResults;
