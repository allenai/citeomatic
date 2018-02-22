import React from 'react'
import pdfImage from './images/pdf.jpg'
import quoteImage from './images/quote.jpg'
import buttonCloseImage from './images/button_close-24.png'

class PaperResults extends React.Component {
  render() {
    return (
      <div>
        <tr>
          <td>Towards Biologically Plausible Deep Learning</td>
          <td>Mattieu Devin, Quoc V. Le</td>
          <td>2015</td>
          <td>
            <div className="progress">
              <div className="progress-bar" role="progressbar" aria-valuenow="70" aria-valuemin="0" aria-valuemax="100" style={{"width":"70%"}}>
                <span className="sr-only">70% Complete</span>
              </div>
            </div>
          </td>
          <td><img src={pdfImage} height="20" width="20" alt="pdf" /></td>
          <td><img src={quoteImage} height="20" width="20" alt="quote" /></td>
          <td><img src={buttonCloseImage} height="20" width="20" alt="not relevant" /></td>
        </tr>
        <tr>
          <td>Towards Biologically Plausible Deep Learning</td>
          <td>Mattieu Devin, Quoc V. Le</td>
          <td>2015</td>
          <td>
            <div className="progress">
              <div className="progress-bar" role="progressbar" aria-valuenow="30" aria-valuemin="0" aria-valuemax="100" style={{"width":"30%"}}>
                <span className="sr-only">30% Complete</span>
              </div>
            </div>
          </td>
          <td><img src={pdfImage} height="20" width="20" alt="pdf" /></td>
          <td><img src={quoteImage} height="20" width="20" alt="quote" /></td>
          <td><img src={buttonCloseImage} height="20" width="20" alt="not relevant" /></td>
        </tr>
      </div>
    );
  }
}

export default PaperResults;
