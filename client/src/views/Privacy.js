import React from 'react';

export default class Privacy extends React.Component {
  render() {
    return (
    <div className="privacy-container centered">
      <div className="privacy">
        <h3 className="intro">Citeomatic Privacy Policy</h3>
        Citeomatic <a href="https://www.semanticscholar.org/privacy-policy">
          follows the privacy policy for Semantic Scholar</a>.

        <p className="tagline your-paper">How we use your paper</p>
        <p className="privacy-text">
        Citeomatic requires a PDF or abstract of a paper to provide citation recommendations.
        For debugging purposes, we may log metadata about the uploaded paper (
          title, abstract, authors and references).  We do not log or store the full text of any
          uploaded paper.
        </p>

        <p className="privacy-text">
        PDFs that are uploaded or fetched from an external URL are not stored persistently.  PDFs
        are discarded after metadata has been extracted from them.
        </p>
        <p className="privacy-text">
        To faciliate sharing results, we save a cache of predictions, which includes the title,
        authors, abstract and bibliography information extracted from a paper.  We may inspect this
        cache at the request of a user or to facilitate debugging.
        </p>
      </div>
    </div>
    );

  }
}
Privacy.contextTypes = {
  router: React.PropTypes.object.isRequired
}
