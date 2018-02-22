import React from 'react';

//import Form from './UploadFileModal';

class YourPaper extends React.Component {
  render() {
    return (
      <div>
        <div className="container your-paper">
          <div className="col-md-4">
            <h3 className="result-header">Your Paper</h3>
          </div>
          <div className="col-md-8">
            <table>
              <tbody>
                  { this.props.sourceFile && 
                  <tr>
                    <td><p className="your-label">File</p></td>
                    <td><p>{this.props.sourceFile}</p></td>
                  </tr>
                  }
                <tr>
                  <td><p className="your-label">Title</p></td>
                  <td><p>{this.props.title}</p></td>
                </tr>
                <tr>
                  <td><p className="your-label">Year</p></td>
                  <td><p>{this.props.year}</p></td>
                </tr>
                <tr>
                  <td><p className="your-label">Authors</p></td>
                  <td><p>{this.props.authors.join(', ')}</p></td>
                </tr>
                <tr>
                  <td className="abstract"><p className="your-label">Abstract</p></td>
                  <td><p>{this.props.abstract}</p></td>
                </tr>
             </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }
}

YourPaper.propTypes = {
  sourceFile: React.PropTypes.string,
  title: React.PropTypes.string.isRequired,
  year: React.PropTypes.number.isRequired,
  abstract: React.PropTypes.string.isRequired,
  authors: React.PropTypes.array.isRequired,
  numCitations: React.PropTypes.number.isRequired
}

export default YourPaper;
