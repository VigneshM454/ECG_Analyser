import { module, test } from 'qunit';
import { setupTest } from 'client/tests/helpers';

module('Unit | Route | record', function (hooks) {
  setupTest(hooks);

  test('it exists', function (assert) {
    let route = this.owner.lookup('route:record');
    assert.ok(route);
  });
});
