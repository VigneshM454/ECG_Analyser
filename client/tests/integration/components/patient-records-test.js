import { module, test } from 'qunit';
import { setupRenderingTest } from 'client/tests/helpers';
import { render } from '@ember/test-helpers';
import { hbs } from 'ember-cli-htmlbars';

module('Integration | Component | patient-records', function (hooks) {
  setupRenderingTest(hooks);

  test('it renders', async function (assert) {
    // Set any properties with this.set('myProperty', 'value');
    // Handle any actions with this.set('myAction', function(val) { ... });

    await render(hbs`<PatientRecords />`);

    assert.dom().hasText('');

    // Template block usage:
    await render(hbs`
      <PatientRecords>
        template block text
      </PatientRecords>
    `);

    assert.dom().hasText('template block text');
  });
});
