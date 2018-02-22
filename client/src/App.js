import React from 'react';

import {Router, IndexRoute, Route, browserHistory} from 'react-router';

import Index from './views/Index';
import Demo from './views/Demo';
import Results from './views/Results';
import Privacy from './views/Privacy';

import 'bootstrap3/dist/css/bootstrap.css';
import 'bootstrap3/dist/css/bootstrap-theme.css';
import './App.css';

import s2Logo from './images/semantic-scholar.light.262x50.png';

function MainLayout({ children }) {
    return (
      <div className="App container">
        <div className="App-header">
          <div className="masthead">
            <ul className="nav nav-pills pull-right">
              <li role="presentation"><a href="/citeomatic">Home</a></li>
              <li role="presentation"><a href="/citeomatic/privacy">Privacy</a></li>
              <li role="presentation" className=""><a href="https://labs.semanticscholar.org/" target="_blank">Blog</a></li>
              <li role="presentation"><a href="mailto:feedback@semanticscholar.org?subject=Citeomatic">Contact</a></li>
            </ul>
            <div className="header-logo">
              <a href="http://www.semanticscholar.org" target="_blank">
                <img className="img-logo" alt="semantic scholar" src={s2Logo}/>
              </a>
            </div>
          </div>
          {children}
         </div>
        <div className="centered">
          <p className="disclaimer">Citeomatic is an alpha developed by the <a href="http://www.semanticscholar.org" target="_blank">Semantic Scholar</a> team at <a href="http://allenai.org/" target="_blank">AI2</a>.</p>
        </div>
     </div>
    );
}

const Root = ({store}) => (
  <Router history={browserHistory}>
    <Route path="/citeomatic/demo" component={Demo} />
    <Route path="/citeomatic" component={MainLayout}>
        <IndexRoute component={Index} />
        <Route path="privacy" component={Privacy} />
        <Route path="pdf/:cacheKey" component={Results} />
        <Route path="url/:cacheKey" component={Results} />
        <Route path="form/:cacheKey" component={Results} />
    </Route>
  </Router>
);

export default Root;
